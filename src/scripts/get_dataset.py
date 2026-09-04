from datasets import load_dataset

ds = load_dataset("EleutherAI/hendrycks_math", "geometry")

ds["train"].to_json("train.jsonl", orient="records", lines=True)
ds["test"].to_json("test.jsonl", orient="records", lines=True)