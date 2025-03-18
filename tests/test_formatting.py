import pytest
from jinja2.exceptions import SecurityError
from instructor.templating import handle_templating
from instructor import Mode


def test_handle_insecure_template():
    with pytest.raises(SecurityError):
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": "{{ self.__init__.__globals__.__builtins__.__import__('os').system('ls') }} {{ variable }}",
                }
            ]
        }
        context = {"variable": "test"}
        handle_templating(kwargs, Mode.TOOLS, context)


def test_handle_templating_with_context():
    kwargs = {"messages": [{"role": "user", "content": "Hello {{ name }}!"}]}
    context = {"name": "Alice"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {"messages": [{"role": "user", "content": "Hello Alice!"}]}


def test_handle_templating_without_context():
    kwargs = {"messages": [{"role": "user", "content": "Hello {{ name }}!"}]}

    result = handle_templating(kwargs, Mode.TOOLS)

    assert result == kwargs


def test_handle_templating_with_anthropic_format():
    kwargs = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello {{ name }}!"}]}
        ]
    }
    context = {"name": "Bob"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello Bob!"}]}
        ]
    }


def test_handle_templating_with_mixed_content():
    kwargs = {
        "messages": [
            {"role": "user", "content": "Hello {{ name }}!"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Nice to meet you, {{ name }}!"}],
            },
        ]
    }
    context = {"name": "Charlie"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "messages": [
            {"role": "user", "content": "Hello Charlie!"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Nice to meet you, Charlie!"}],
            },
        ]
    }


def test_handle_templating_with_secret_context():
    from pydantic import BaseModel, SecretStr

    class UserContext(BaseModel):
        name: str
        address: SecretStr

    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": "{{ user.name }}'s address is '{{ user.address.get_secret_value() }}'",
            }
        ]
    }
    context = {
        "user": UserContext(
            name="Jason", address=SecretStr("123 Secret St, Hidden City")
        )
    }

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "messages": [
            {
                "role": "user",
                "content": "Jason's address is '123 Secret St, Hidden City'",
            }
        ]
    }

    # Ensure the original SecretStr is not exposed when rendered
    assert str(context["user"].address) == "**********"


def test_handle_templating_with_cohere_format():
    kwargs = {
        "message": "Hello {{ name }}!",
        "chat_history": [{"message": "Previous message to {{ name }}"}],
    }
    context = {"name": "David"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "message": "Hello David!",
        "chat_history": [{"message": "Previous message to David"}],
    }


def test_handle_templating_with_gemini_format():
    kwargs = {
        "contents": [
            {"role": "user", "parts": ["Hello {{ name }}!", "How are you {{ name }}?"]}
        ]
    }
    context = {"name": "Eve"}

    result = handle_templating(kwargs, Mode.TOOLS, context)

    assert result == {
        "contents": [{"role": "user", "parts": ["Hello Eve!", "How are you Eve?"]}]
    }
