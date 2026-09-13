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
from utils import ROOT, load_user_prompt, evaluate_model

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
logger = logging.getLogger(__name__)
DEVICE = "cuda:0"

class GRPODataset(Dataset):
    def __init__(self, data_path, prompt_template):
        self.data = []
        with xopen(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                prompt = prompt_template.format(question=item["problem"])
                self.data.append((prompt, item["solution"]))
        logger.info("Loaded %d GRPO samples", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch):
    prompts, answers = zip(*batch)
    return list(prompts), list(answers)

def sequence_log_probs(model, input_ids, response_mask):
    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:].float()
    token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)

def encode_rollouts(tokenizer, prompts, responses, max_length):
    rows, masks = [], []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-max_length:]
        response_budget = max_length - len(prompt_ids)
        response_ids = response_ids[:response_budget]
        ids = prompt_ids + response_ids
        rows.append(ids)
        masks.append([0] * len(prompt_ids) + [1] * len(response_ids))
    width = max(len(row) for row in rows)
    input_ids = [row + [tokenizer.pad_token_id] * (width - len(row)) for row in rows]
    response_masks = [mask + [0] * (width - len(mask)) for mask in masks]
    return (
        torch.tensor(input_ids, dtype=torch.long, device=DEVICE),
        torch.tensor(response_masks, dtype=torch.long, device=DEVICE),
    )

def generate_group(llm, prompts, group_size, sampling_max_tokens, sampling_min_tokens, sampling_temperature):
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [output.text.strip() for result in outputs for output in result.outputs]

def train(
    model_path: Path,
    data_path: Path,
    generate_path: Path,
    n_grpo_steps: int,
    learning_rate: float,
    advantage_eps: float,
    rollout_batch_size: int,
    group_size: int,
    sampling_temperature: float,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    epochs_per_rollout_batch: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    gpu_memory_utilization: float,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=str(model_path),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=1024,
        max_num_seqs=1,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0"
    )

    policy.train()
    for p in policy.parameters():
        p.requires_grad_(False)
    for block in policy.model.layers[-2:]:
        for p in block.parameters():
            p.requires_grad_(True)
    for p in policy.lm_head.parameters():
        p.requires_grad_(True)

    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    dataloader = DataLoader(
        GRPODataset(data_path, load_user_prompt()),
        batch_size=rollout_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    step = 0
    while step < n_grpo_steps:
        for prompts, targets in dataloader:
            if step >= n_grpo_steps:
                break

            prompt_lengths = [
                len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                for prompt in prompts
            ]
            prompts, targets = zip(*[
                (prompt, target)
                for prompt, target, prompt_length in zip(
                    prompts, targets, prompt_lengths
                )
                if prompt_length + sampling_max_tokens <= 1024
            ]) if any(
                prompt_length + sampling_max_tokens <= 1024
                for prompt_length in prompt_lengths
            ) else ([], [])
            prompts = list(prompts)
            targets = list(targets)
            if len(prompts) == 0:
                continue

            responses = generate_group(
                llm,
                prompts,
                group_size,
                sampling_max_tokens,
                sampling_min_tokens,
                sampling_temperature,
            )
            repeated_prompts = [prompt for prompt in prompts for _ in range(group_size)]

            metrics = [
                r1_zero_reward_fn(
                    response, targets[index // group_size]
                )
                for index, response in enumerate(responses)
            ]
            rewards = torch.tensor([
                metric["reward"] for metric in metrics
            ], dtype=torch.float32, device=DEVICE)
            grouped = rewards.view(-1, group_size)
            advantages = (
                grouped - grouped.mean(dim=1, keepdim=True)
            ).reshape(-1)
            advantages = advantages / (
                grouped.std(dim=1).repeat_interleave(group_size) + advantage_eps
            )
            input_ids, response_mask = encode_rollouts(
                tokenizer, repeated_prompts, responses, 1024
            )
            old_log_probs = []
            with torch.no_grad():
                for start in range(0, len(responses), train_batch_size):
                    end = start + train_batch_size
                    old_log_probs.append(
                        sequence_log_probs(
                            policy,
                            input_ids[start:end],
                            response_mask[start:end],
                        )
                    )
            old_log_probs = torch.cat(old_log_probs)
            accumulated_loss = 0.0
            loss_steps = 0
            for _ in range(epochs_per_rollout_batch):
                for start in range(0, len(responses), train_batch_size):
                    end = start + train_batch_size
                    current_log_probs = sequence_log_probs(
                        policy,
                        input_ids[start:end],
                        response_mask[start:end],
                    )
                    ratios = torch.exp(
                        current_log_probs - old_log_probs[start:end]
                    )
                    unclipped = ratios * advantages[start:end]
                    clipped = ratios.clip(0.8, 1.2) * advantages[start:end]
                    loss = -torch.minimum(unclipped, clipped).mean()
                    (loss / gradient_accumulation_steps).backward()
                    accumulated_loss += loss.item()
                    loss_steps += 1
            step += 1
            if step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            logger.info(
                "step=%d loss=%.6f reward=%.4f format_reward=%.4f "
                "answer_reward=%.4f correct=%d/%d reward_groups=%s",
                step,
                accumulated_loss / loss_steps,
                rewards.mean().item(),
                sum(metric["format_reward"] for metric in metrics) / len(metrics),
                sum(metric["answer_reward"] for metric in metrics) / len(metrics),
                sum(metric["answer_reward"] for metric in metrics),
                len(metrics),
                ",".join(
                    "".join(str(int(reward)) for reward in group)
                    for group in rewards.view(-1, group_size).tolist()
                ),
            )
    policy.save_pretrained(generate_path)
    tokenizer.save_pretrained(generate_path)

def main(
    model_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-DPO"),
    data_path: Path = typer.Option(ROOT / "data" / "MATH" / "original" / "train.jsonl"),
    test_path:     Path  = typer.Option(ROOT / "data" / "MATH" / "original" / "test.jsonl"),
    generate_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-GRPO"),
    output_path:   Path  = typer.Option(ROOT / "results" / "GRPO.jsonl"),
    n_grpo_steps: int = typer.Option(200),
    learning_rate: float = typer.Option(1e-5),
    advantage_eps: float = typer.Option(1e-6),
    rollout_batch_size: int = typer.Option(1),
    group_size: int = typer.Option(2),
    sampling_temperature: float = typer.Option(1.0),
    sampling_min_tokens: int = typer.Option(4),
    sampling_max_tokens: int = typer.Option(256),
    epochs_per_rollout_batch: int = typer.Option(2),
    train_batch_size: int = typer.Option(1),
    gradient_accumulation_steps: int = typer.Option(8),
    gpu_memory_utilization: float = typer.Option(0.7),
    temperature:   float = typer.Option(1.0),
    max_tokens:    int   = typer.Option(1024),
):
    logging.basicConfig(
        filename="logs/grpo.log",
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    # train(
    #     model_path,
    #     data_path,
    #     generate_path,
    #     n_grpo_steps,
    #     learning_rate,
    #     advantage_eps,
    #     rollout_batch_size,
    #     group_size,
    #     sampling_temperature,
    #     sampling_min_tokens,
    #     sampling_max_tokens,
    #     epochs_per_rollout_batch,
    #     train_batch_size,
    #     gradient_accumulation_steps,
    #     gpu_memory_utilization,
    # )
    evaluate_model(
        model_path=generate_path,
        data_path=test_path,
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )

if __name__ == "__main__":
    typer.run(main)
