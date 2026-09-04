# Tech Challenge

This repository aims to reproduce a simplified procedure of *post-training* on `Qwen2.5-Math-1.5B`.

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