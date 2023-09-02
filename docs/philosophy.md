# Philosophy

The philosophy behind this library is to provide a **lightweight** and **flexible** approach to leveraging language models (LLMs) to do **structured output without imposing unnecessary dependencies or abstractions.**

The `instructor` library serves as a bridge from text-based language model interaction to Object-Oriented Programming, seamlessly integrating LLMs into the programming paradigms we're familiar with. By treating LLMs as callable functions that return typed objects, `instructor` demystifies their complexity, making them more accessible for everyday projects. This approach maintains the flexibility and power of Python, letting you write custom code without unnecessary constraints.

1. Define a Schema `#!python class StructuredData(BaseModel):`
2. Encapsulate all your LLM logic into a function `#!python def extract(a) -> StructuredData:` 
3. Define typed computations against your data with `#!python def compute(data: StructuredData):`

Please note that the library is designed to be adaptable and open-ended, allowing you to customize and extend its functionality based on your specific requirements.

If you have any further questions or ideas hit me up on [twitter](https://twitter.com/jxnlco)
