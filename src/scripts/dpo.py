"""
DPO.
"""

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import typer

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen

from drgrpo_grader import r1_zero_reward_fn
from utils import ROOT, load_test_data, load_user_prompt, evaluate_model

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

logger = logging.getLogger(__name__)

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        with xopen(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                prompt = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
                chosen = tokenizer(item["chosen"], add_special_tokens=False)["input_ids"]
                reject = tokenizer(item["rejected"], add_special_tokens=False)["input_ids"]
                if len(prompt) + len(chosen) <= 1024 and len(prompt) + len(reject) <= 1024:
                    self.data.append((prompt, chosen, reject))
        logger.info("Loaded %d DPO samples.", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch, tokenizer):
    prompts, chosens, rejects = zip(*batch)
    chosen_ids = [p + c for p, c in zip(prompts, chosens)]
    reject_ids = [p + r for p, r in zip(prompts, rejects)]
    chosen_masks = [[0] * len(p) + [1] * len(c) for p, c in zip(prompts, chosens)]
    reject_masks = [[0] * len(p) + [1] * len(r) for p, r in zip(prompts, rejects)]

    def pad(rows, masks):
        length = max(len(row) for row in rows)
        return (torch.tensor([row + [tokenizer.pad_token_id] * (length - len(row)) for row in rows]),
                torch.tensor([mask + [0] * (length - len(mask)) for mask in masks]))

    chosen_ids, chosen_masks = pad(chosen_ids, chosen_masks)
    reject_ids, reject_masks = pad(reject_ids, reject_masks)
    return chosen_ids, chosen_masks, reject_ids, reject_masks

def seq_log_prob(model, input_ids, response_mask, device):
    input_ids = input_ids.to(device)
    response_mask = response_mask.to(device)
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:].float()
    token_logprobs = F.log_softmax(logits, dim = -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(dim=-1)

def dpo_train(model_path, data_path, generate_path, num_epochs, num_layers, beta):
    policy =AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0"
    )
    reference =AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy.train()
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    for p in policy.parameters():
        p.requires_grad_(False)
    for block in policy.model.layers[-num_layers:]:
        for p in block.parameters():
            p.requires_grad_(True)
    for p in policy.lm_head.parameters():
        p.requires_grad_(True)

    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr = 5e-6)
    loader = DataLoader(
        DPODataset(data_path, tokenizer),
        batch_size=1, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    accumulation = 8
    for epoch in range(num_epochs):
        remain_steps = 0
        for idx, batch in enumerate(tqdm(loader, desc=f"dpo epoch {epoch}")):
            chosen, chosen_mask, reject, reject_mask = batch
            ref_chosen = seq_log_prob(reference, chosen, chosen_mask, "cpu")
            ref_reject = seq_log_prob(reference, reject, reject_mask, "cpu")
            policy_chosen = seq_log_prob(policy, chosen, chosen_mask, "cuda:0")
            policy_reject = seq_log_prob(policy, reject, reject_mask, "cuda:0")
            loss = -F.sigmoid(beta * (policy_chosen - ref_chosen.to("cuda:0")) - (policy_reject - ref_reject.to("cuda:0"))).mean()
            (loss / accumulation).backward()
            remain_steps += 1
            if(idx + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                remain_steps = 0
                logger.info("epoch=%d step=%d loss=%.6f", epoch, idx, loss.item())
        if remain_steps:
            optimizer.step()
            optimizer.zero_grad()
    policy.save_pretrained(generate_path)
    tokenizer.save_pretrained(generate_path)

def main(
    model_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-RSFT"),
    data_path:     Path  = typer.Option(ROOT / "data" / "MATH" / "dpo" / "Math-Step-DPO-10K.jsonl"),
    generate_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-DPO"),
    beta: int = typer.Option(0.4),
    num_epochs: int = typer.Option(1),
    num_layers: int = typer.Option(2),
    output_path:   Path  = typer.Option(ROOT / "results" / "DPO.jsonl"),
    temperature:   float = typer.Option(1.0),
    max_tokens: int = typer.Option(1024),
):
    logging.basicConfig(filename="logs/dpo.log",
                        level=logging.INFO,
                        format="%(name)s - %(levelname)s - %(message)s",
    )
    # dpo_train(model_path, data_path, generate_path, num_epochs, num_layers, beta)
    evaluate_model(model_path=generate_path, data_path=ROOT / "data" / "MATH" / "original" / "test.jsonl", output_path=output_path,temperature=temperature, max_tokens=max_tokens)

if __name__ == "__main__":
    typer.run(main)