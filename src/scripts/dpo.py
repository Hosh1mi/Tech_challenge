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
from utils import ROOT, evaluate_model

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

logger = logging.getLogger(__name__)

MAX_LENGTH = 2048
DEVICE = "cuda:0"

class DPODataset(Dataset):
    def __init__(self, data_path, cache_path, tokenizer):
        self.data = []
        with xopen(data_path, "r") as data_file, xopen(cache_path, "r") as cache_file:
            for line in data_file:
                item = json.loads(line)
                prompt = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
                chosen = tokenizer(item["chosen"], add_special_tokens=False)["input_ids"]
                reject = tokenizer(item["rejected"], add_special_tokens=False)["input_ids"]
                if len(prompt) + len(chosen) <= MAX_LENGTH and len(prompt) + len(reject) <= MAX_LENGTH:
                    cache = json.loads(next(cache_file))
                    self.data.append((prompt, chosen, reject, cache["ref_chosen_logprob"], cache["ref_reject_logprob"]))
        logger.info("Loaded %d DPO samples.", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch, tokenizer):
    prompts, chosens, rejects, ref_chosens, ref_rejects = zip(*batch)
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
    return chosen_ids, chosen_masks, reject_ids, reject_masks, torch.tensor(ref_chosens), torch.tensor(ref_rejects)

def seq_log_prob(model, input_ids, response_mask):
    input_ids = input_ids.to(DEVICE)
    response_mask = response_mask.to(DEVICE)
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:].float()
    token_logprobs = F.log_softmax(logits, dim = -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(dim=-1)

def cache_reference_logprobs(model_path, data_path, cache_path):
    reference =AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with xopen(data_path, "r") as data_file, xopen(cache_path, "w") as cache_file, torch.inference_mode():
        for line in tqdm(data_file, desc="cache reference log-probs"):
            item = json.loads(line)
            prompt = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
            chosen = tokenizer(item["chosen"], add_special_tokens=False)["input_ids"]
            reject = tokenizer(item["rejected"], add_special_tokens=False)["input_ids"]
            if len(prompt) + len(chosen) > MAX_LENGTH or len(prompt) + len(reject) > MAX_LENGTH:
                continue
            chosen_ids = torch.tensor([prompt + chosen])
            reject_ids = torch.tensor([prompt + reject])
            chosen_mask = torch.tensor([[0] * len(prompt) + [1] * len(chosen)])
            reject_mask = torch.tensor([[0] * len(prompt) + [1] * len(reject)])
            ref_chosen = seq_log_prob(reference, chosen_ids, chosen_mask).item()
            ref_reject = seq_log_prob(reference, reject_ids, reject_mask).item()
            cache_file.write(json.dumps({"ref_chosen_logprob": ref_chosen, "ref_reject_logprob": ref_reject}) + "\n")
    del reference
    torch.cuda.empty_cache()

def dpo_train(model_path, data_path, cache_path, generate_path, num_epochs, num_layers, beta):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = DPODataset(data_path, cache_path, tokenizer)
    policy =AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=DEVICE
    )
    policy.train()
    for p in policy.parameters():
        p.requires_grad_(False)
    for block in policy.model.layers[-num_layers:]:
        for p in block.parameters():
            p.requires_grad_(True)
    for p in policy.lm_head.parameters():
        p.requires_grad_(True)

    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=5e-6)
    loader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    accumulation = 8
    for epoch in range(num_epochs):
        remain_steps = 0
        accumulated_loss = 0.0
        accumulated_margin = 0.0
        accumulated_accuracy = 0.0
        for idx, batch in enumerate(tqdm(loader, desc=f"dpo epoch {epoch}")):
            chosen, chosen_mask, reject, reject_mask, ref_chosen, ref_reject = batch
            policy_chosen = seq_log_prob(policy, chosen, chosen_mask)
            policy_reject = seq_log_prob(policy, reject, reject_mask)
            reward_margin = beta * ((policy_chosen - ref_chosen.to(DEVICE)) - (policy_reject - ref_reject.to(DEVICE)))
            loss = -F.logsigmoid(reward_margin).mean()
            (loss / accumulation).backward()
            remain_steps += 1
            accumulated_loss += loss.item()
            accumulated_margin += reward_margin.mean().item()
            accumulated_accuracy += (reward_margin > 0).float().mean().item()
            if(idx + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                remain_steps = 0
                logger.info("epoch=%d step=%d loss=%.6f margin=%.6f preference_accuracy=%.4f", epoch, idx, accumulated_loss / accumulation, accumulated_margin / accumulation, accumulated_accuracy / accumulation)
                accumulated_loss = 0.0
                accumulated_margin = 0.0
                accumulated_accuracy = 0.0
        if remain_steps:
            optimizer.step()
            optimizer.zero_grad()
            logger.info("epoch=%d step=%d loss=%.6f margin=%.6f preference_accuracy=%.4f", epoch, idx, accumulated_loss / remain_steps, accumulated_margin / remain_steps, accumulated_accuracy / remain_steps)
    policy.save_pretrained(generate_path)
    tokenizer.save_pretrained(generate_path)

def main(
    model_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-RSFT"),
    data_path:     Path  = typer.Option(ROOT / "data" / "MATH" / "dpo" / "Math-Step-DPO-10K.jsonl"),
    cache_path:    Path  = typer.Option(ROOT / "data" / "MATH" / "dpo" / "Math-Step-DPO-10K-cache.jsonl"),
    generate_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-DPO"),
    beta: float = typer.Option(0.35),
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
    cache_reference_logprobs(model_path, data_path, cache_path)
    dpo_train(model_path, data_path, cache_path, generate_path, num_epochs, num_layers, beta)
    evaluate_model(model_path=generate_path, data_path=ROOT / "data" / "MATH" / "original" / "test.jsonl", output_path=output_path,temperature=temperature, max_tokens=max_tokens)

if __name__ == "__main__":
    typer.run(main)
