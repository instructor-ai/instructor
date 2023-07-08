from openai_function_call.dsl import messages as m
from openai_function_call.dsl.messages import system as s


def test_create_message():
    assert m.Message(
        role=m.MessageRole.SYSTEM,
        content="Hello, world!",
    ).dict() == {
        "role": "system",
        "content": "Hello, world!",
    }


def test_create_user_message():
    assert m.UserMessage(
        content="Hello, world!",
    ).dict() == {
        "role": "user",
        "content": "Hello, world!",
    }


def test_create_system_message():
    assert m.SystemMessage(content="I am nice").dict() == {
        "role": "system",
        "content": "I am nice",
    }


def test_assistance_message():
    assert m.AssistantMessage(content="I am nice").dict() == {
        "role": "assistant",
        "content": "I am nice",
    }


def test_create_tagged_message():
    assert m.TaggedMessage(content="I am nice", tag="data").dict() == {
        "role": "user",
        "content": "<data>I am nice</data>",
    }


def test_task_message():
    assert s.SystemTask(task="task").dict() == {
        "role": "system",
        "content": f"You are a world class state of the art algorithm capable of correctly completing the following task: `task`.",
    }


def test_chain_of_thought_message():
    assert m.ChainOfThought().dict() == {
        "role": "assistant",
        "content": "Lets think step by step to get the correct answer:",
    }
