import json
import logging

from pathlib import Path
from vllm import LLM, SamplingParams
from xopen import xopen

from drgrpo_grader import r1_zero_reward_fn

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "MATH"

logger = logging.getLogger(__name__)

def load_test_data(data_path: Path = TEST_PATH) -> list[dict]:
    test_files = Path(data_path).glob("*/test.jsonl")

    examples = []
    for test_file in test_files:
        with xopen(test_file) as file:
            for line in file:
                example = json.loads(line)
                examples.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "level": example.get("level"),
                    "type": example.get("type"),
                })

    logger.info("Loaded %d MATH test cases from %s", len(examples), data_path)
    return examples

def load_user_prompt(prompt_path: Path = ROOT / "src" / "prompts" / "user.prompt") -> str:
    with open(prompt_path, encoding="utf-8") as file:
        return file.read().strip()

def build_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    prompts = [prompt_template.format(question=example["question"]) for example in examples]
    logger.info("Formatted %d prompts", len(prompts))
    return prompts

def analyze_result_categories(metrics: list[dict[str, float]]) -> dict[str, int]:
    categories = {
        "Correct (format=1, answer=1)": 0,
        "Format correct, answer wrong (format=1, answer=0)": 0,
        "Format incorrect (format=0, answer=0)": 0,
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
            assert(0)
    return categories

def extract_answer(solution: str) -> str:
    """
    Note that every solution has a `\\boxed{}` as answer.
    So we use that as the answer.
    """
    target = r"\boxed{"
    pos = solution.rfind(target)

    # Haven't find one that hasn't got this feature
    # if pos == -1:
    #     return solution.strip()

    start = pos + len(target)
    end = start

    while end < len(solution):
        end += 1

    return target + solution[start:end]


def convert_dataset(input_path: Path, output_path: Path) -> None:
    with xopen(input_path, "r") as fin, xopen(output_path, "w") as fout:
        for line in fin:
            item = json.loads(line)
            prompt = load_user_prompt().format(question=item["problem"])

            answer = item["solution"].strip() + "\n</think>\n<answer>" + extract_answer(item["solution"]) + "</answer>"

            fout.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "answer": answer
                    },
                )
                + "\n"
            )

def evaluate_model(
    model_path:             Path,
    output_path:            Path,
    data_path:              Path  = TEST_PATH,
    prompt_path:            Path  = ROOT / "src" / "prompts" / "user.prompt",
    temperature:            float = 1.0,
    max_tokens:             int   = 1024,
    gpu_memory_utilization: float = 0.7,
) -> dict[str, float]:
    examples = load_test_data(data_path)
    prompts = build_prompts(examples, load_user_prompt(prompt_path))

    logger.info("Initializing vLLM model from %s", model_path)
    llm = LLM(
        model=str(model_path), 
        gpu_memory_utilization=gpu_memory_utilization
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0, 
        max_tokens=max_tokens,
        stop=["</answer>"], 
        include_stop_str_in_output=True,
    )
    raw_responses = llm.generate(prompts, sampling_params)
    responses = [response.outputs[0].text.strip() for response in raw_responses]

    metrics = []
    results = []
    for index, (example, response) in enumerate(zip(examples, responses)):
        metric = r1_zero_reward_fn(response, example["answer"])
        metrics.append(metric)
        results.append({
            "id": index, 
            "type": example["type"], 
            "level": example["level"],
            "question": example["question"], "model_response": response,
            "answer": example["answer"], "metrics": metric,
        })

    accuracy = {key: sum(metric[key] for metric in metrics) / len(metrics) for key in metrics[0]}
    for key, value in sorted(accuracy.items()):
        logger.info("%s: %.4f", key, value)
    for category, count in analyze_result_categories(metrics).items():
        logger.info("%s: %d/%d (%.1f%%)", category, count, len(metrics), count / len(metrics) * 100)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with xopen(output_path, "w") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    return accuracy
