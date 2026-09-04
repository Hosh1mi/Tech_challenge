import logging
import os
from pathlib import Path

import typer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

from utils import TEST_PATH, ROOT, evaluate_model


def main(
    model_path: Path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B"),
    data_path: Path = typer.Option(TEST_PATH),
    output_path: Path = typer.Option(ROOT / "results" / "baseline.jsonl"),
    temperature: float = typer.Option(1.0),
    max_tokens: int = typer.Option(1024),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    evaluate_model(
        model_path=model_path,
        data_path=data_path,
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    typer.run(main)
