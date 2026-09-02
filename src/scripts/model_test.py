import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYS_PROMPT = """
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
```{instruction}```

# Answer:
```
"""

USR_PROMPT = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""

instruction = "Solve this math task."

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

MODEL_PATH = "models/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# connect the 2 prompts or separate roles???

prompt = SYS_PROMPT.format(instruction=instruction) + "\n" + SYS_PROMPT.format(instruction=instruction)
# messages = [
#     {
#         "role": "system",
#         "content": SYS_PROMPT.format(instruction=instruction)
#     },
#     {
#         "role": "user",
#         "content": USR_PROMPT.format(question=question)
#     }
# ]

# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

if __name__ == "__main__":
    llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.7)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1,
        max_tokens=1024
    )

    outputs = llm.generate(
        [prompt],
        sampling_params
    )

    for output in outputs:
        print(output.outputs[0].text)

    print("=== DONE ===")

    del llm