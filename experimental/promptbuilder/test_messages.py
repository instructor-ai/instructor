from prompt_builder import *


def test_create_message():
    assert Message(
        role=MessageRole.SYSTEM,
        content="Hello, world!",
    ).dict() == {
        "role": "system",
        "content": "Hello, world!",
    }


def test_create_user_message():
    assert UserMessage(
        content="Hello, world!",
    ).dict() == {
        "role": "user",
        "content": "Hello, world!",
    }


def test_create_system_message():
    assert SystemMessage(content="I am nice").dict() == {
        "role": "system",
        "content": "I am nice",
    }


def test_assistance_message():
    assert AssistantMessage(content="I am nice").dict() == {
        "role": "assistant",
        "content": "I am nice",
    }


def test_create_tagged_message():
    assert TaggedMessage(content="I am nice", tag="data").dict() == {
        "role": "user",
        "content": "<data>I am nice</data>",
    }


def test_expert_system_message():
    assert ExpertSystem(task="task").dict() == {
        "role": "system",
        "content": "You are a world class, state of the art agent capable of correctly completing the task: `task`",
    }


def test_chain_of_thought_message():
    assert ChainOfThought().dict() == {
        "role": "assistant",
        "content": "Lets think step by step to get the correct answer:",
    }
