---
title: "Example Ordering"
description: "The order in which examples are added into the prompt can affect LLM outputs"
---

# Example Ordering

The order in which examples are added into the prompt can affect LLM outputs<sup><a href="https://arxiv.org/abs/2104.08786">1</a><a href="https://arxiv.org/abs/2106.01751">2</a><a href="https://arxiv.org/abs/2101.06804">3</a><a href="https://aclanthology.org/2022.naacl-main.191/">4</a></sup>.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())

class Output(BaseModel):
    code: str

prediction = client.chat.completions.create(
    model="gpt-4o",
    response_model=Output,
    messages=[
        {"role": "user", "content": f"""
                            Write javascript to calculate a factorial. Do not include comments.

                            Example 1:
                                function functionName(parameters) {{
                                    // function body
                                    return value;
                                }}

                            Example 2:
                                const functionName = (parameters) => {{
                                    // function body
                                    return value;
                                }};
                            """
        }
    ]
)

prediction_examples_switched = client.chat.completions.create( # (1)!
    model="gpt-4o",
    response_model=Output,
    messages=[
        {"role": "user", "content": f"""
                            Write javascript to calculate a factorial. Do not include comments.

                            Example 1:
                                const functionName = (parameters) => {{
                                    // function body
                                    return value;
                                }};

                            Example 2:
                                function functionName(parameters) {{
                                    // function body
                                    return value;
                                }}
                            """
        }
    ]
)

print(prediction.code) # (2)!
"""
const factorial = (n) => {
    if (n < 0) return -1;
    if (n === 0) return 1;
    return n * factorial(n - 1);
};
"""
print(prediction_examples_switched.code) # (3)!
"""
function factorial(n) {
    if (n === 0 || n === 1) return 1;
    return n * factorial(n - 1);
}
"""
```

1.  The order of the examples is switched in this prompt (arrow function, then function declaration)
2.  The LLM outputs an arrow function
3.  The LLM outputs a function declaration

### References

<sup id="ref-1">1</sup>: [Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity](https://arxiv.org/abs/2104.08786)

<sup id="ref-1">2</sup>: [Reordering Examples Helps during Priming-based Few-Shot Learning](https://arxiv.org/abs/2106.01751)

<sup id="ref-1">3</sup>: [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)

<sup id="ref-1">4</sup>: [Learning To Retrieve Prompts for In-Context Learning](https://aclanthology.org/2022.naacl-main.191/)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
