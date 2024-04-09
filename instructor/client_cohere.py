import cohere
import instructor

from typing import overload
from typing import Type
from pydantic import BaseModel
from .utils import extract_python_from_codeblock


@overload
def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs,
) -> instructor.Instructor: ...


def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs,
) -> instructor.Instructor:
    assert (
        mode
        in {
            instructor.Mode.COHERE_TOOLS,
        }
    ), "Mode be one of {instructor.Mode.COHERE_TOOLS}"

    assert isinstance(
        client, (cohere.Client)
    ), "Client must be an instance of cohere.Cohere"

    def create_cohere_structured_output(**kwargs):
        """
        (1) generates a tool snippet for a pydantic object
        (2) formats a tool use prompt with the tool snippet
        (3) sends the tool use prompt to cohere client chat
        """
        if "messages" in kwargs and "response_model" in kwargs:
            conversation = kwargs.pop("messages")
            response_model = kwargs.pop("response_model")
            tool_snippet = generate_tool_snippet_pydantic(client, response_model, **kwargs)
            tool_use_prompt = format_tool_use_prompt(conversation, tool_snippet)
            return client.chat(message=tool_use_prompt, **kwargs)
        elif "messages" in kwargs:
            print(kwargs)
            conversation = kwargs.pop("messages")
            message = conversation[-1]["content"]
            chat_history = conversation[:-1]
            return client.chat(message=message, chat_history=chat_history, **kwargs)
        return client.chat(**kwargs)

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create_cohere_structured_output, mode=mode),
        provider=instructor.Provider.COHERE,
        mode=mode,
        **kwargs,
    )


def prompt_model_for_python_class_def(pydantic_class: BaseModel) -> str:
    """returns a prompt to generate python code that defines the provided pydantic object"""
    json_schema = pydantic_class.model_json_schema()
    class_name = pydantic_class.__name__
    prompt = f"""\
Carefully read the following JSON schema and write a Python class definition that matches the schema.
{json_schema}

The pydantic schema was obtained by running:
```python
schema = {class_name}.model_json_schema()
```

Please write the Python code to define the class {class_name} that matches the schema.
Do not include any imports or additional code (e.g. `from pydantic import BaseModel`).

The class should have the following structure:
- The class should be named {class_name}
- The class should inherit from BaseModel
- The class should have fields the following fields {', '.join(pydantic_class.__fields__.keys())}
- should have the correct type and description as specified in the schema above.

If you need to define any other classes, please do so above the definition of `{class_name}`.

Your output should start with ```python\nclass"""  # noqa
    return prompt


def generate_tool_snippet_pydantic(client: cohere.Client, response_model: Type[BaseModel], **kwargs) -> str:
    """Returns a string representing Python code to define a tool which takes pydantic object as a parameter."""
    param_name = response_model.__name__.lower()
    tool_name = f"extract_{param_name}"
    tool_description = f"Extracts a valid {response_model.__name__} object based on the conversation history"

    prompt = prompt_model_for_python_class_def(response_model)
    new_kwargs = {k: v for k, v in kwargs.items() if k not in ["messages", "response_model"]}
    response = client.chat(message=prompt, **new_kwargs).text
    class_definitions = extract_python_from_codeblock(response)

    tool_snippet = f"```python\n{class_definitions}\n\ndef {tool_name}({param_name}: {response_model.__name__}):\n"
    if tool_description:
        tool_snippet += f'    """{tool_description}\n\n'
    tool_snippet += '    Args:\n'
    tool_snippet += f'        {param_name} ({response_model.__name__}): {response_model.__name__} object\n'
    tool_snippet += '    """\n'
    tool_snippet += '    pass\n```'
    return tool_snippet


def render_chat_history(_conversation: list[dict]) -> str:
    """Renders chat history as a string."""
    chat_hist_str = ""
    for turn in _conversation:
        chat_hist_str += "<|START_OF_TURN_TOKEN|>"
        if turn['role'] == 'user':
            chat_hist_str += "<|USER_TOKEN|>"
        elif turn['role'] == 'assistant':
            chat_hist_str += "<|CHATBOT_TOKEN|>"
        else: # role == system
            chat_hist_str += "<|SYSTEM_TOKEN|>"
        chat_hist_str += turn['content']
    chat_hist_str += "<|END_OF_TURN_TOKEN|>"
    return chat_hist_str


def format_tool_use_prompt(conversation: list, tool_snippet: str) -> str:
    """Returns tool use prompt with code for tools being provided by `tool_snippet`."""
    preamble = """\
<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

## Available Tools
Here is a list of tools that you have available to you:
"""
    instructions = """\
<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
```json
[
    {
        "tool_name": title of the tool in the specification,
        "parameters": input into the tool as defined in the specs, or {} if it takes no parameters
    }
]```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
"""
    chat_history = render_chat_history(conversation)
    tool_use_prompt = f"{preamble}\n{tool_snippet}{chat_history}\n{instructions}"
    return tool_use_prompt
