"""Claude Agent SDK integration for Instructor.

This module provides integration with the Claude Agent SDK, enabling structured
outputs using instructor's familiar interface with the Claude Agent SDK's
agentic capabilities.

Example:
    ```python
    from instructor import from_claude_agent_sdk
    from pydantic import BaseModel
    import anyio

    class User(BaseModel):
        name: str
        age: int

    async def main():
        client = from_claude_agent_sdk()

        user = await client.create(
            response_model=User,
            messages=[{"role": "user", "content": "Extract: John is 25 years old"}]
        )
        print(user.name)  # John
        print(user.age)   # 25

    anyio.run(main)
    ```
"""

from .client import from_claude_agent_sdk, ClaudeAgentSDKClient, claude_agent_sdk_create
from .utils import handle_claude_agent_sdk, reask_claude_agent_sdk

__all__ = [
    "from_claude_agent_sdk",
    "ClaudeAgentSDKClient",
    "claude_agent_sdk_create",
    "handle_claude_agent_sdk",
    "reask_claude_agent_sdk",
]
