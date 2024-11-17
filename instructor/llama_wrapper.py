"""Wrapper for llama-cpp-python to make it compatible with instructor."""
from typing import Any, Dict, List, Optional, Union, Generator
from llama_cpp import Llama
import json

class LlamaWrapper:
    """Wrapper for llama-cpp-python that implements a chat-like interface."""

    def __init__(self, llm: Llama):
        self.llm = llm
        self.chat = self.ChatCompletions(llm)

    class ChatCompletions:
        def __init__(self, llm: Llama):
            self.llm = llm
            self.completions = self

        def create(
            self,
            messages: List[Dict[str, str]],
            response_model: Any = None,
            max_tokens: int = 100,
            temperature: float = 0.1,
            stream: bool = False,
            tools: Optional[List[Dict]] = None,
            tool_choice: Optional[Dict] = None,
            **kwargs
        ) -> Union[Dict, Generator]:
            """Create a chat completion that mimics OpenAI's interface."""

            # Filter out unsupported parameters
            supported_params = {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stream': stream
            }

            # Add any other supported parameters from kwargs
            for key in ['top_p', 'stop', 'frequency_penalty', 'presence_penalty']:
                if key in kwargs:
                    supported_params[key] = kwargs[key]

            # Convert chat messages to prompt
            prompt = self._convert_messages_to_prompt(messages)

            # If tools are provided, add function calling context
            if tools:
                tool_spec = tools[0]["function"]  # We only support one tool for now
                prompt = (
                    f"{prompt}\n\n"
                    f"Extract the information and respond in the following JSON format:\n"
                    f"{json.dumps(tool_spec['parameters'], indent=2)}\n"
                )

            try:
                if stream:
                    return self._stream_completion(prompt, **supported_params)
                else:
                    return self._create_completion(prompt, **supported_params)
            except Exception as e:
                raise Exception(f"Error in llama completion: {str(e)}")

        def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
            """Convert chat messages to a prompt string."""
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            return "\n".join(prompt_parts)

        def _create_completion(
            self, prompt: str, **kwargs
        ) -> Dict:
            """Create a completion and format response like OpenAI's API."""
            try:
                response = self.llm.create_completion(
                    prompt=prompt,
                    **kwargs
                )

                return {
                    "id": response.get("id", ""),
                    "object": "chat.completion",
                    "created": response.get("created", 0),
                    "model": response.get("model", "llama"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response["choices"][0]["text"].strip()
                        },
                        "finish_reason": response["choices"][0].get("finish_reason", "stop")
                    }],
                    "usage": response.get("usage", {})
                }
            except Exception as e:
                raise Exception(f"Error in completion: {str(e)}")

        def _stream_completion(
            self, prompt: str, **kwargs
        ) -> Generator:
            """Create a streaming completion."""
            try:
                stream = self.llm.create_completion(
                    prompt=prompt,
                    **kwargs
                )

                if not isinstance(stream, Generator):
                    # If streaming is not supported, yield a single chunk
                    yield {
                        "choices": [{
                            "delta": {
                                "content": stream["choices"][0]["text"]
                            },
                            "finish_reason": stream["choices"][0].get("finish_reason")
                        }]
                    }
                    return

                for chunk in stream:
                    if isinstance(chunk, dict) and "choices" in chunk:
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": chunk["choices"][0]["text"]
                                },
                                "finish_reason": chunk["choices"][0].get("finish_reason")
                            }]
                        }
                    else:
                        # Handle raw text chunks
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": str(chunk)
                                },
                                "finish_reason": None
                            }]
                        }

            except Exception as e:
                raise Exception(f"Error in streaming completion: {str(e)}")
