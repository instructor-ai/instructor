from typing import Any, Iterator, Optional, Union, List, TypedDict, Literal

class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class FunctionCall(TypedDict):
    name: str
    arguments: str

class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCall

class CompletionDict(TypedDict, total=False):
    text: str
    finish_reason: Optional[str]
    usage: Optional[Usage]
    function_call: Optional[FunctionCall]
    tool_calls: Optional[List[ToolCall]]

class CompletionChunk:
    text: str
    finish_reason: Optional[str]
    usage: Optional[Usage]
    function_call: Optional[FunctionCall]
    tool_calls: Optional[List[ToolCall]]

    def __init__(
        self,
        text: str = "",
        finish_reason: Optional[str] = None,
        usage: Optional[Usage] = None,
        function_call: Optional[FunctionCall] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.usage = usage
        self.function_call = function_call
        self.tool_calls = tool_calls

    def to_dict(self) -> CompletionDict:
        """Convert to dictionary format"""
        result: CompletionDict = {
            "text": self.text
        }
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        if self.usage is not None:
            result["usage"] = self.usage
        if self.function_call is not None:
            result["function_call"] = self.function_call
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        return result

class Completion:
    text: str
    finish_reason: Optional[str]
    usage: Usage
    function_call: Optional[FunctionCall]
    tool_calls: Optional[List[ToolCall]]

    def __init__(
        self,
        text: str = "",
        finish_reason: Optional[str] = None,
        usage: Optional[Usage] = None,
        function_call: Optional[FunctionCall] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.function_call = function_call
        self.tool_calls = tool_calls

    def to_dict(self) -> CompletionDict:
        """Convert to dictionary format"""
        result: CompletionDict = {
            "text": self.text,
            "usage": self.usage  # Usage is always present due to default in __init__
        }
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        if self.function_call is not None:
            result["function_call"] = self.function_call
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        return result

    def get_dict(self) -> CompletionDict:
        """Get dictionary representation for compatibility"""
        return self.to_dict()

class Llama:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        **kwargs: Any
    ) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.kwargs = kwargs

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Create a completion."""
        raise NotImplementedError()

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        raise NotImplementedError()

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize tokens."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset the model state."""
        raise NotImplementedError()
