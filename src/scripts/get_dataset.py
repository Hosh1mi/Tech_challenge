from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main")

dataset["train"].to_json("train.jsonl", orient="records", lines=True)
dataset["test"].to_json("test.jsonl", orient="records", lines=True)