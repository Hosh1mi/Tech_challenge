"""
RSFT.
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

def sample_dataset(
    model_path: Path,
    source_path: Path, 
    output_path: Path, 
    g: int, 
    max_tokens: int, 
    min_tokens: int, 
    seed: int
):
    examples = load_test_data(source_path)
    prompt_template = load_user_prompt()
    prompts = [prompt_template.format(question=item["question"]) for item in examples]
    llm = LLM(
        model = str(model_path),
        gpu_memory_utilization=0.7
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        n=g,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )
    outputs = llm.generate(prompts, sampling_params)
    with xopen(output_path, "w") as f:
        for example, result in zip(examples, outputs):
            for i in result.outputs:
                response = i.text
                reward = r1_zero_reward_fn(response, example["answer"])
                if reward["answer_reward"] == 1.0:
                    prompt = prompt_template.format(question=example["question"])
                    f.write(json.dumps(
                        {
                            "prompt": prompt,
                            "answer": response
                        },
                        ensure_ascii=False
                    )
                    + "\n"
                )
    logger.info("Samples generated")

class RSFTDataset(Dataset):
    """Same as SFT"""
    def __init__(self, data_path, tokenizer):
        self.data = []
        with xopen(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                prompt_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
                answer_ids = tokenizer(item["answer"], add_special_tokens=False)["input_ids"]
                if len(prompt_ids) + len(answer_ids) <= 1024:
                    self.data.append((prompt_ids, answer_ids))
        logger.info("Loaded %d samples from %s", len(self.data), data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt_ids, answer_ids = self.data[index]

        return{
            "input_ids": prompt_ids + answer_ids,
            "response_mask": [0] * len(prompt_ids) + [1] * len(answer_ids)
        }

def collate_fn(batch, tokenizer):
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

def rsft_train(
    model_path: Path,
    data_path: Path,
    generate_path: Path,
    num_epochs: int
):
    model =AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6
    )
    dataloader = DataLoader(
        RSFTDataset(data_path, tokenizer),
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    accumulation = 8
    for epoch in range(num_epochs):
        accumulated_loss = 0.0
        accumulated_response_entropy = 0.0
        for idx, batch in enumerate(tqdm(dataloader, desc=f"rsft epoch {epoch}")):
            input_ids = batch["input_ids"].to("cuda:0")
            response_mask = batch["response_mask"].to("cuda:0")
            outputs = model(input_ids=input_ids)

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            shift_mask = response_mask[:, 1:].float()
            
            token_cross_entropy = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view_as(shift_mask)

            cross_entropy = (token_cross_entropy * shift_mask).sum() / shift_mask.sum()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            token_entropy = -(probs * log_probs).sum(dim=-1)
            response_entropy = (token_entropy * shift_mask).sum() / shift_mask.sum()

            accumulated_loss += cross_entropy.item()
            accumulated_response_entropy += response_entropy.item()
            (cross_entropy / accumulation).backward()

            if(idx + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                mean_cross_entropy = accumulated_loss / accumulation
                mean_respose_entropy = accumulated_response_entropy / accumulation
                logger.info("epoch=%d step=%d loss=%.6f response_entropy=%.6f", epoch, idx, mean_cross_entropy, mean_respose_entropy)

                accumulated_loss = 0.0
                accumulated_response_entropy = 0.0
    model.save_pretrained(generate_path)
    tokenizer.save_pretrained(generate_path)

def main(
    model_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-SFT"),
    source_path: Path = typer.Option(ROOT / "data" / "MATH" / "original" / "train.jsonl"),
    test_path: Path = typer.Option(ROOT / "data" / "MATH" / "rsft" / "rsft_train.jsonl"),
    data_path:     Path  = typer.Option(ROOT / "data" / "MATH" / "original" / "test.jsonl"),
    generate_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B-RSFT"),
    output_path:   Path  = typer.Option(ROOT / "results" / "RSFT.jsonl"),
    g: int = typer.Option(4),
    temperature:   float = typer.Option(1.0),
    max_tokens: int = typer.Option(1024),
    min_tokens: int = typer.Option(1),
    num_epochs: int = typer.Option(1),
    seed: int = typer.Option(42),
):
    logging.basicConfig(filename="logs/rsft.log",
                        level=logging.INFO,
                        format="%(name)s - %(levelname)s - %(message)s",
    )
    # sample_dataset(model_path, source_path, test_path, g, max_tokens, min_tokens, seed)
    # rsft_train(model_path, test_path, generate_path, num_epochs)
    evaluate_model(
        model_path=generate_path,
        data_path=data_path,
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )

if __name__ == "__main__":
    typer.run(main)  