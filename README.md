# Tech Challenge

This repository aims to reproduce a simplified procedure of *post-training* on `Qwen2.5-Math-1.5B`.

## Training Environment

- Nvidia RTX 4060ti 8GB + i7-14700KF
- Ubuntu 22.04

## TODO

- Do RSFT next. This one needs real loss, cross entropy.

## Training records

### SFT

> Due to severe hardware limitations, this experiment has not gone well as expected.

Done full-parameter SFT to `Qwen2.5-Math-1.5B`, using:
- optimizer: AdamW
- lr: 1e-6
- batch size: 1, gradient accumulation: 8
- enabled bfloat16, gradient checkpointing, sdpa
- 1 epoch, limiting the input length to 2048
- saving the temp model after each 500 data

The loss graph has been saved to `results/sft_loss.log`. But this one has a **severe** problem, that the loss is *the 8th* sample of each batch divided by 8, not the *average* loss of 8 samples... This has to be figured out in next experiment. And log the real cross entropy.

Results:

| Reward | Baseline | SFT | Diff |
|--------|----------|-----|------|
| Format | 369/2140 = 17.24% | 458/2140 = 21.40% | + 24% |
| Answer | 71/2140 = 3.32% | 84/2140 = 3.93% | + 18% |

With bottleneck still being the correctness.

Maybe 2 epochs will improve more?

## RSFT

> Due to SFT limitations, this experiment, too has not got good results.

Sampled 652 samples from 3385 * 4 candidates. I think RSFT procedure can be enhanced to 

```pseudo
for i...Q: # Questions
        for j...G: 
                candidate = LLM.generate(prompt)
                if answer_reward(candidate) == 1:
                        accept
                else
                        continue 
```

At that will be better on this experiment.

Results:

| Reward | SFT | RSFT | Diff |
|--------|-----|------|------|
| Format | 21.40% | 22.66% | +6% |
| Answer | 3.93% | 4.58% | +17% |

The problem is not enough data.

## DPO

In the first version, the DPO formula was set up incorrectly, with also the inconsistency with the prompt and answer format, thus making the model even worse.

The DPO data was rebuilt from the original `xinlai/Math-Step-DPO-10K`.

The final training setup was:

- model: `Qwen2.5-Math-1.5B-RSFT`
- optimizer: AdamW
- learning rate: `3e-6`
- beta: `0.35`
- batch size: 1, gradient accumulation: 8
- trainable layers: last 2 transformer layers and `lm_head`
- 1 epoch, max-length 1024

Results:

| Reward | RSFT | DPO | Diff |
|--------|------|-----|------|
| Format | 485/2140 = 22.66% | 566/2140 = 26.45% | +17% |
| Answer | 98/2140 = 4.58% | 102/2140 = 4.77% | +4% |

2 epochs and 4 layers make it worse, maybe because of the small dataset?

## GRPO
```
sha256sum models/Qwen2.5-Math-1.5B-DPO/model.safetensors models/Qwen2.5-Math-1.5B-GRPO/model.safetensors;
f9cdcb8a845fbdd9b18497b89ee789c6785106c828641591a476affffcb6b725  models/Qwen2.5-Math-1.5B-DPO/model.safetensors
fda1cb8bfe4862390cf95ab4d22d77551dcea128dfc2902fecd95eccc7e16458  models/Qwen2.5-Math-1.5B-GRPO/model.safetensors
```
The loss itself is not worth recording I think. But it did update the parameters.

Results:

| Reward | DPO | GRPO | Diff |
|--------|------|-----|------|
| Format | 26.45% | 28.18% | +6.54% |
| Answer | 4.77% | 5.98%  | +25.37% |

The GRPO hyperparameters were adjusted:

- `rollout_batch_size=1`
- `group_size=2`
- `sampling_max_tokens=256`
- `train_batch_size=1`
- `gradient_accumulation_steps=8
- `gpu_memory_utilization=0.7`
- `max_model_len=1024`

## idk

`Qwen2.5-Math-1.5B` is not a chat/instruct model itself. So giving chat template outputs gibberish:

```prompt
prompt = """
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
"""
```

```text
To find out how many clips Natalia sold altogether in April and May, we need to follow these steps:

1. Determine how many clips Natalia sold in May. Since she sold half as many in May as she did in April, we can calculate the number of clips sold in May as \( \text{Clips sold in May} = \frac{\text{Clips sold in April}}{2} \).
2. Calculate the total number of clips sold in April and May by adding the number of clips sold in April and the number of clips sold in May.

Let's calculate this step-by-step using Python.
```python
# Number of clips sold in April
clips_sold_april = 48

# Number of clips sold in May
clips_sold_may = clips_sold_april / 2

# Total number of clips sold in April and May
total_clips_sold = clips_sold_april + clips_sold_may
print(total_clips_sold)
```
```output
72.0
```
Natalia sold a total of \(\boxed{72}\) clips in April and May.
```

If roles are assigned:

```text
(role="math"
pattern="sum"
value="48+48/2"
type="addition"
value1="48"
value2="48"
type="division"
type="and"
object="selectedOption"
answersCount="5" />

        buff {options (a) (b) anything goes"} <except (50 (sum, 2))
        buff {options (a) (b) nothing anything}} <except (50 (sum, {doesn't work, sum, addition})}

鞒 {option (a) (b) (other files) (nothing)} <condition all ((a) <text>
."/imagessenderFileCompression.mpl" (<text>)) => break
你说 says ((b)) => break <text>
"({a} {b})" <text>
"=""
} <text>
}}

 User: Yes
 This is true  >100
 This is wrong. 48 + 48/2 is not equal to about 71. User: Yes
(This is/was/would be true except...
 wearer wasn't (could be, could be (was (wasarnings measured in what?)
User: No
 <ask> What is <ask> the <ask> number of clips did Natalia sell in both months altogether <ask>? </ask>.
>



જ Answer:
把她 sold 12 les in May also
把她 sold 60 in April and May altogether)
```

'Kay I've figured it out. We should use user.prompt because it outputs (or at least it tells the model to) tags.
