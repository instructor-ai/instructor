from typing import Dict, Any, Iterator, Optional, Union, List

class CompletionChunk:
    text: str
    finish_reason: Optional[str]

class Completion:
    text: str
    finish_reason: Optional[str]
    usage: Dict[str, int]

class Llama:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        **kwargs: Any
    ) -> None: ...

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Completion, Iterator[CompletionChunk]]: ...

    def tokenize(self, text: str) -> List[int]: ...
    def detokenize(self, tokens: List[int]) -> str: ...
    def reset(self) -> None: ...
