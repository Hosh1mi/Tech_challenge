"""
SFT training.
This procedure updates ALL parameters.
Expectation: Format correctness increases. Not sure about Answer yet.
"""

import logging
import os
import torch
import json
import torch.nn.functional as F

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from xopen import xopen

import typer
from tqdm import tqdm

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

from utils import ROOT, evaluate_model, convert_dataset 
logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with xopen(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                prompt_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
                answer_ids = tokenizer(item["answer"], add_special_tokens=False)["input_ids"]
                if len(prompt_ids) + len(answer_ids) <= 2048:
                    self.data.append((prompt_ids, answer_ids))
        logger.info("Loaded %d samples from %s", len(self.data), data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt_ids, answer_ids = self.data[index]
        input_ids = prompt_ids + answer_ids

        response_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

        return{
            "input_ids": input_ids,
            "response_mask": response_mask
        }

def load_model(
    model_path: Path = ROOT / "models" / "Qwen2.5-Math-1.5B"
):
    # Use transformers to train, vLLM to test
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa", # failed to install flash-attn, SAD :(
        device_map="cuda:0"
    )
    return model

def load_tokenizer(
    model_path: Path = ROOT / "models" / "Qwen2.5-Math-1.5B"
):
    return AutoTokenizer.from_pretrained(model_path)

def collate_fn(batch, tokenizer):
    """
    [
        [10, 20, 30, 40, pad, pad],
        [11, 21, 31, 41, 51, 61]
    ]
    """
    max_length = max(
        len(item["input_ids"])
        for item in batch
    )

    input_ids = []
    response_masks = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["response_mask"]

        padding_length = max_length - len(ids)

        ids = ids + [tokenizer.pad_token_id] * padding_length
        mask = mask + [0] * padding_length

        input_ids.append(ids)
        response_masks.append(mask)

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long
        ),
        "response_mask": torch.tensor(
            response_masks,
            dtype=torch.long
        )
    }

def sft_train(
    num_epochs: int,
    generate_path: Path
):

    model = load_model()
    tokenizer = load_tokenizer()

    model.gradient_checkpointing_enable()   # Trade Memory with time
    model.config.use_cache = False          # Disable KV Cache, useless when teaching

    # convert_dataset(ROOT / "data" / "MATH" / "original" / "train.jsonl", ROOT / "data" / "MATH" / "sft" / "sft_train.jsonl")
    # convert_dataset(ROOT / "data" / "MATH" / "original" / "test.jsonl", ROOT / "data" / "MATH" / "sft" / "sft_test.jsonl")

    # idk which one to choose really. SGD surely is smallest
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
    )

    logger.info("Optimizer set up")
    device = "cuda:0"
    gradient_accumulation_steps = 8
    for epoch in range(num_epochs):
        logger.info(f"epoch {epoch} started")
        for shard_path in sorted((ROOT / "data" / "MATH" / "sft").glob("sft_train_*.jsonl")):
            train_dataset = SFTDataset(shard_path, tokenizer)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, tokenizer)
            )
            for idx, batch in enumerate(tqdm(train_dataloader, desc=f"epoch {epoch} {shard_path.stem}")):
                input_ids = batch["input_ids"].to(device)
                response_mask = batch["response_mask"].to(device)

                outputs = model(input_ids=input_ids)

                logits = outputs.logits[:, :-1, :]
                labels = input_ids[:, 1:]
                shift_mask = response_mask[:, 1:].float()

                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="none",
                )

                losses = losses.view_as(shift_mask)
                loss = (losses * shift_mask).sum() / shift_mask.sum()

                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.info(
                        f"epoch={epoch}, "
                        f"step={idx}, "
                        f"loss={loss.item():.6f}"
                    )

                if (idx + 1) % 500 == 0:
                    checkpoint_path = generate_path / f"checkpoint-step-{idx + 1}"
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    logger.info("Saved checkpoint to %s", checkpoint_path)


    model.save_pretrained(generate_path)
    tokenizer.save_pretrained(generate_path)

def main(
    model_path:    Path  = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B"),
    generate_path: Path  = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-SFT"),
    data_path:     Path  = typer.Option(ROOT / "data" / "MATH" / "original" / "test.jsonl"),
    output_path:   Path  = typer.Option(ROOT / "results" / "SFT.jsonl"),
    temperature:   float = typer.Option(1.0),
    max_tokens:    int   = typer.Option(1024), # Not sure enough or not
    num_epochs:    int   = typer.Option(1)
) -> None:
    logging.basicConfig(
        filename="logs/sft.log",
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Started training")
    sft_train(num_epochs, generate_path)
    evaluate_model(
        model_path=generate_path,
        data_path=data_path,
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )

if __name__ == "__main__":
    typer.run(main)
