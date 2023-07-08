# Using the Chatcompletion

To get started with this api we must first instantiate a `ChatCompletion` object and build the api call
by piping messages and functions to it.

::: openai_function_call.dsl.completion

## Messages Types

The basis of a message is defined as a `dataclass`. However we provide helper functions and classes that provide additional functionality in the form of templates. 

::: openai_function_call.dsl.messages.base

## Helper Messages / Templates

::: openai_function_call.dsl.messages.messages

::: openai_function_call.dsl.messages.user