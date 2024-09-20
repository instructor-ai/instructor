import pytest
from textwrap import dedent
from instructor.patch import handle_templating
from jinja2 import Template


def test_handle_templating_with_context():
    messages = [{"role": "user", "content": "Hello {{ name }}!"}]
    context = {"name": "Alice"}

    result = handle_templating(messages, context)

    assert result == [{"role": "user", "content": "Hello Alice!"}]


def test_handle_templating_without_context():
    messages = [{"role": "user", "content": "Hello {{ name }}!"}]

    result = handle_templating(messages)

    assert result == messages


def test_handle_templating_with_anthropic_format():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello {{ name }}!"}]}
    ]
    context = {"name": "Bob"}

    result = handle_templating(messages, context)

    assert result == [
        {"role": "user", "content": [{"type": "text", "text": "Hello Bob!"}]}
    ]


def test_handle_templating_with_mixed_content():
    messages = [
        {"role": "user", "content": "Hello {{ name }}!"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Nice to meet you, {{ name }}!"}],
        },
    ]
    context = {"name": "Charlie"}

    result = handle_templating(messages, context)

    assert result == [
        {"role": "user", "content": "Hello Charlie!"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Nice to meet you, Charlie!"}],
        },
    ]


def test_handle_templating_with_secret_context():
    from pydantic import BaseModel, SecretStr

    class UserContext(BaseModel):
        name: str
        address: SecretStr

    messages = [
        {
            "role": "user",
            "content": "{{ user.name }}'s address is '{{ user.address.get_secret_value() }}'",
        }
    ]
    context = {"user": UserContext(name="Jason", address="123 Secret St, Hidden City")}

    result = handle_templating(messages, context)

    assert result == [
        {"role": "user", "content": "Jason's address is '123 Secret St, Hidden City'"}
    ]

    # Ensure the original SecretStr is not exposed when rendered
    assert str(context["user"].address) == "**********"


def test_handle_templating_with_loop():
    messages = [
        {
            "role": "user",
            "content": dedent(
                """
                Here are the items:
                {% for item in items %}
                  - {{ item.name }}: ${{ item.price }}
                {% endfor %}
                Total: ${{ total }}
            """
            ),
        }
    ]
    context = {
        "items": [
            {"name": "Apple", "price": 0.5},
            {"name": "Banana", "price": 0.3},
            {"name": "Orange", "price": 0.7},
        ],
        "total": 1.5,
    }

    result = handle_templating(messages, context)

    expected_content = dedent(
        """
        Here are the items:

          - Apple: $0.5

          - Banana: $0.3

          - Orange: $0.7
          
        Total: $1.5
        """
    )

    assert result[0]["role"] == "user"
    assert result[0]["content"].strip() == expected_content.strip()
