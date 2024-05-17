# Philosophy

The instructor values [simplicity](https://eugeneyan.com/writing/simplicity/) and flexibility in leveraging language models (LLMs). It offers a streamlined approach for structured output, avoiding unnecessary dependencies or complex abstractions. Let [Pydantic](https://docs.pydantic.dev/latest/) do the heavy lifting.

> “Simplicity is a great virtue but it requires hard work to achieve it and education to appreciate it. And to make matters worse: complexity sells better.” — Edsger Dijkstra

### Proof that its simple

1. Most users will only need to learn `response_model` and `patch` to get started.
2. No new prompting language to learn, no new abstractions to learn.

### Proof that its transparent

1. We write very little prompts, and we don't try to hide the prompts from you.
2. We'll do better in the future to give you config over the 2 prompts we do write, Reasking and JSON_MODE prompts.

### Proof that its flexible

1. If you build a system with OpenAI directly, it is easy to incrementally adopt instructor.
2. Add `response_model` and if you want to revert, just remove it.

## The zen of `instructor`

Maintain the flexibility and power of Python, without unnecessary constraints.

Begin with a function and a return type hint – simplicity is key. With my experience maintaining a large enterprize framework at my previous job over many years I've learned that the goal of a making a useful framework is minimizing regret, both for the author and hopefully for the user.

1. Define a Schema `#!python class StructuredData(BaseModel):`
2. Define validators and methods on your schema.
3. Encapsulate all your LLM logic into a function `#!python def extract(a) -> StructuredData:`
4. Define typed computations against your data with `#!python def compute(data: StructuredData):` or call methods on your schema `#!python data.compute()`

It should be that simple.

## My Goals

The goal for the library, [documentation](https://jxnl.github.io/instructor/), and [blog](https://jxnl.github.io/instructor/blog/), is to help you be a better python programmer and as a result a better AI engineer.

- The library is a result of my desire for simplicity.
- The library should help maintain simplicity in your codebase.
- I won't try to write prompts for you,
- I don't try to create indirections or abstractions that make it hard to debug in the future

Please note that the library is designed to be adaptable and open-ended, allowing you to customize and extend its functionality based on your specific requirements. If you have any further questions or ideas hit me up on [twitter](https://twitter.com/jxnlco)

Cheers!
