---
description: "DECOMP involves using a LLM to break down a complicated task into sub tasks that it has been provided with"
---

Decomposed Prompting<sup><a href="https://arxiv.org/pdf/2210.02406">1</a></sup> leverages a Language Model (LLM) to deconstruct a complex task into a series of manageable sub-tasks. Each sub-task is then processed by specific functions, enabling the LLM to handle intricate problems more effectively and systematically.

In the code snippet below, we define a series of data models and functions to implement this approach.

The `derive_action_plan` function generates an action plan using the LLM, which is then executed step-by-step. Each action can be

1. InitialInput: Which represents the chunk of the original prompt we need to process
2. Split : An operation to split strings using a given separator
3. StrPos: An operation to help extract a string given an index
4. Merge: An operation to join a list of strings together using a given character

We can implement this using `instructor` as seen below.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Union, Optional

client = instructor.from_openai(OpenAI())


class InitialInput(BaseModel):
    input_data: str


class Split(BaseModel):
    split_char: str = Field(
        description="""This is the character to split
        the string with"""
    )


class StrPos(BaseModel):
    index: int = Field(
        description="""This is the index of the character
        we wish to return"""
    )


class Merge(BaseModel):
    merge_char: str = Field(
        description="""This is the character to merge the
        inputs we plan to pass to this function with"""
    )


class Action(BaseModel):
    id: int = Field(
        description="""Unique Incremental id to identify
        this action with"""
    )
    action: Union[Split, StrPos, Merge, InitialInput]
    input_source: Optional[int] = Field(
        description="""Prior Action Id whose inputs we
        wish to use for this action""",
        default=None,
    )


class ActionPlan(BaseModel):
    plan: list[Action]


def derive_action_plan(task_description: str) -> ActionPlan:
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """Generate an action plan to help you complete
                the task outlined by the user""",
            },
            {"role": "user", "content": task_description},
        ],
        response_model=ActionPlan,
        max_retries=3,
        model="gpt-4o",
    )


if __name__ == "__main__":
    task = """Concatenate the second letter of every word in Jack
    Ryan together"""
    plan = derive_action_plan(task)
    print(plan.model_dump_json(indent=2))
    """
    {
      "plan": [
        {
          "id": 1,
          "action": {
            "input_data": "Jack Ryan"
          },
          "input_source": null
        },
        {
          "id": 2,
          "action": {
            "split_char": " "
          },
          "input_source": 1
        },
        {
          "id": 3,
          "action": {
            "index": 1
          },
          "input_source": 2
        },
        {
          "id": 4,
          "action": {
            "merge_char": ""
          },
          "input_source": 3
        }
      ]
    }
    """

    curr = ""
    cache = {}

    for action in plan.plan:
        input_source = action.input_source
        if input_source:
            input_source = cache[input_source]  # type:ignore

        if isinstance(action.action, InitialInput):
            curr = action.action.input_data
            cache[action.id] = curr
        elif isinstance(action.action, Split):
            curr = input_source.split(action.action.split_char)  # type:ignore
            cache[action.id] = curr  # type:ignore
        elif isinstance(action.action, StrPos):
            assert isinstance(input_source, list)
            extracted_strings = [
                value[action.action.index] for value in input_source
            ]  # type:ignore
            cache[action.id] = extracted_strings
        elif isinstance(action.action, Merge):
            curr = action.action.merge_char.join(input_source)  # type:ignore
            cache[action.id] = curr

    print(curr)
    #> ay
```

### References

<sup id="ref-1">1</sup>: [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/pdf/2210.02406)
