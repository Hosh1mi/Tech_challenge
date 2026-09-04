"""
Baseline evaluation script for GSM8K test set.
"""

import os
import typer
import logging
import json

from pathlib import Path
from vllm import LLM, SamplingParams

from xopen import xopen
from drgrpo_grader import r1_zero_reward_fn

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

def load_sys_prompt() -> str:
    path = ROOT / "src" / "prompts" / "system.prompt"
    with open(path, 'r') as f:
        return f.read().strip()

def load_usr_prompt() -> str:
    path = ROOT / "src" / "prompts" / "user.prompt"
    with open(path, 'r') as f:
        return f.read().strip()

# def build_prompt(tokenizer, instruction, question):
#     system = load_sys_prompt().format(instruction=instruction)
#     user = load_usr_prompt().format(question=question)
#     messages = [
#         {
#             "roles": "system",
#             "content": system
#         },
#         {
#             "roles": "user",
#             "content": user
#         }
#     ]
#     return tokenizer.apply_chat_template(
#         messages,
#     )

def build_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    prompts = []
    for example in examples:
        formatted_prompt = prompt_template.format(question=example["question"])
        prompts.append(formatted_prompt)
    
    logger.info(f"Formatted {len(prompts)} prompts")
    return prompts

def load_test_data(datapath = ROOT / "data" / "gsm8k" / "test.jsonl") -> list[dict]:
    examples = []

    with xopen(datapath) as f:
        for line in f:
            examples.append(json.loads(line))

    logger.info(f"===Successfully loaded {len(examples)} testcases from {datapath}")
    return examples

def analyze_result_categories(metrics: list[dict[str, float]]) -> dict[str, int]:
    categories = {
        "Correct (format=1, answer=1)": 0,
        "Format correct, answer wrong (format=1, answer=0)": 0, 
        "Format incorrect (format=0, answer=0)": 0
    }
    
    for metric in metrics:
        format_reward = metric.get("format_reward", 0)
        answer_reward = metric.get("answer_reward", 0)
        
        if format_reward == 1.0 and answer_reward == 1.0:
            categories["Correct (format=1, answer=1)"] += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            categories["Format correct, answer wrong (format=1, answer=0)"] += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            categories["Format incorrect (format=0, answer=0)"] += 1
        else:
            logger.warning(f"Unexpected reward combination: format={format_reward}, answer={answer_reward}")
    
    return categories

def main(
    model_path = typer.Option(ROOT / "models" / "Qwen2.5-Math-1.5B"),
    data_path = typer.Option(ROOT / "data" / "gsm8k" / "test.jsonl"),
    output_path = typer.Option(ROOT / "results" / "baseline.jsonl"),
    temperature : float = typer.Option(1.0),
    max_tokens : int = typer.Option(1024),
):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=== Loading prompts ===")
    prompt = load_usr_prompt()

    logger.info("=== Loading MATH test data ===")
    test_examples = load_test_data(data_path)

    prompts = build_prompts(test_examples, prompt)

    answers = [example["answer"] for example in test_examples]

    logger.info("=== Initializing vLLM model ===")

    llm = LLM(
        model=model_path, 
        gpu_memory_utilization=0.7,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    logger.info("=== Generating responses ===")

    responses_raw = llm.generate(prompts, sampling_params)
    responses = []
    for r in responses_raw:
        responses.append(r.outputs[0].text.strip())

    logger.info("=== Successfully generated responses ===")

    metrics = []
    results = []

    for i, (prompt, response, answer) in enumerate(zip(prompts, responses, answers)):
        metr = r1_zero_reward_fn(response, answer)
        metrics.append(metr)

        result = {
            "id": i,
            "model_response": response,
            "answer": answer,
            "metrics": metr
        }
        results.append(result)

    accuracy = {}
    for key in metrics[0].keys(): # "format_reward", "answer_reward", "reward"
        metric_vals = []
        for m in metrics:
            metric_vals.append(m[key])
        tot = 0
        for val in metric_vals:
            tot = tot + val
        average = tot / len(metric_vals)
        accuracy[key] = average
    
    category_counts = analyze_result_categories(metrics)

    logger.info("=== Evaluation Results ===")
    for key, value in sorted(accuracy.items()):
        logger.info(f"{key}: {value:.4f}")
    
    logger.info("=== Analysing results ===")
    for category, count in category_counts.items():
        percentage = (count / len(metrics)) * 100
        logger.info(f"{category}: {count}/{len(metrics)} ({percentage:.1f}%)")

    logger.info("=== Saving results ===")
    with xopen(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info("=== Baseline evaluated ===")

    
if __name__ == "__main__":
    typer.run(main)