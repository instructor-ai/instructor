# Using the Prompt Pipeline

To use the Prompt Pipeline in OpenAI Function Call, you need to instantiate a `ChatCompletion` object and build the API call by piping messages and functions to it.

## The ChatCompletion Object

The `ChatCompletion` object is the starting point for constructing your API call. It provides the necessary methods and attributes to define the conversation flow and include function calls.

::: instructor.dsl.completion

## Messages Types

The basis of a message is defined as a `dataclass`. However, we provide helper functions and classes that provide additional functionality in the form of templates. 

::: instructor.dsl.messages.base

## Helper Messages / Templates

::: instructor.dsl.messages.messages

::: instructor.dsl.messages.user