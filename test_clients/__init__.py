from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, Iterator, List, Optional, TypeVar, Union

try:
    from llama_cpp import Llama
    from llama_cpp.llama_types import CompletionChunk, Completion
except ImportError:
    pass  # Types will be imported during runtime in LlamaWrapper.__init__

import instructor
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class Choice:
    """A choice in a completion response"""
    delta: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    message: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result: Dict[str, Any] = {
            "index": self.index,
        }
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs
        if self.delta:
            if self.tool_calls:
                self.delta["tool_calls"] = self.tool_calls
            result["delta"] = self.delta
        if self.message:
            result["message"] = self.message
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result

class OpenAIResponse:
    """Base class for OpenAI API responses"""
    def __init__(
        self,
        id: str = None,
        created: int = None,
        model: str = None,
        object_type: str = None,
        choices: List[Choice] = None,
        usage: Dict[str, int] = None,
    ):
        """Initialize the response

        Args:
            id: Response ID
            created: Timestamp when response was created
            model: Model name
            object_type: Response object type
            choices: List of choices
            usage: Token usage statistics
        """
        self._id = id
        self._created = created
        self._model = model
        self._object = object_type
        self._choices = choices or []
        self._usage = usage or {}

    @property
    def id(self):
        """Get response ID"""
        return self._id

    @property
    def created(self):
        """Get creation timestamp"""
        return self._created

    @property
    def model(self):
        """Get model name"""
        return self._model

    @property
    def object(self):
        """Get object type"""
        return self._object

    @property
    def choices(self):
        """Get list of choices"""
        return self._choices

    @choices.setter
    def choices(self, value):
        """Set list of choices"""
        self._choices = value

    @property
    def usage(self):
        """Get token usage statistics"""
        return self._usage

    @usage.setter
    def usage(self, value):
        """Set token usage statistics"""
        self._usage = value

    def to_dict(self):
        """Convert response to dictionary"""
        return {
            "id": self.id,
            "created": self.created,
            "model": self.model,
            "object": self.object,
            "choices": [choice.to_dict() for choice in self.choices],
            "usage": self.usage
        }

    def __getattr__(self, name):
        """Get attribute from dictionary representation"""
        try:
            return self.to_dict()[name]
        except KeyError:
            raise AttributeError(f"'OpenAIResponse' object has no attribute '{name}'")

class StreamingResponse(OpenAIResponse):
    """Response from a streaming completion request"""
    def __init__(self, chunk=None, **kwargs):
        """Initialize the streaming response

        Args:
            chunk: Response chunk from llama.cpp
            **kwargs: Additional arguments to pass to OpenAIResponse
        """
        # Extract text and metadata from chunk if not provided in kwargs
        if 'choices' not in kwargs and chunk is not None:
            if isinstance(chunk, dict):
                if 'choices' in chunk:
                    # Handle llama-cpp response format
                    choice = chunk['choices'][0]
                    text = choice.get('text', '')
                    finish_reason = choice.get('finish_reason')
                else:
                    # Handle raw dict format
                    text = chunk.get('text', '')
                    finish_reason = chunk.get('finish_reason')
            else:
                text = getattr(chunk, 'text', '')
                finish_reason = getattr(chunk, 'finish_reason', None)

            # Set choices with the extracted text
            kwargs['choices'] = [
                Choice(
                    index=0,
                    delta={"role": "assistant", "content": text},
                    finish_reason=finish_reason
                )
            ]

        # Initialize with required OpenAI response fields
        super().__init__(
            id=kwargs.pop('id', f"chatcmpl-{hash(str(chunk))& 0xFFFFFFFF:08x}"),
            created=kwargs.pop('created', int(datetime.now().timestamp())),
            model=kwargs.pop('model', "llama"),
            object_type=kwargs.pop('object_type', "chat.completion.chunk"),
            **kwargs
        )

    def __iter__(self):
        """Return self as iterator"""
        return self

    def __next__(self):
        """Get next streaming response"""
        raise StopIteration

class CompletionResponse(OpenAIResponse):
    """Response from a completion request"""
    def __init__(self, chunk=None, **kwargs):
        """Initialize the completion response

        Args:
            chunk: Response chunk from llama.cpp
            **kwargs: Additional arguments to pass to OpenAIResponse
        """
        # Extract text and metadata from chunk if not provided in kwargs
        if 'choices' not in kwargs and chunk is not None:
            if isinstance(chunk, dict):
                if 'choices' in chunk:
                    # Handle llama-cpp response format
                    choice = chunk['choices'][0]
                    text = choice.get('text', '')
                    finish_reason = choice.get('finish_reason')
                else:
                    # Handle raw dict format
                    text = chunk.get('text', '')
                    finish_reason = chunk.get('finish_reason')
            else:
                text = getattr(chunk, 'text', '')
                finish_reason = getattr(chunk, 'finish_reason', None)

            # Set choices with the extracted text
            kwargs['choices'] = [
                Choice(
                    index=0,
                    message={"role": "assistant", "content": text},
                    finish_reason=finish_reason
                )
            ]

        # Initialize with required OpenAI response fields
        super().__init__(
            id=kwargs.pop('id', f"chatcmpl-{hash(str(chunk))& 0xFFFFFFFF:08x}"),
            created=kwargs.pop('created', int(datetime.now().timestamp())),
            model=kwargs.pop('model', "llama"),
            object_type=kwargs.pop('object_type', "chat.completion"),
            **kwargs
        )

    def get_dict(self):
        """Get dictionary representation of response"""
        return self.to_dict()

class LlamaWrapper:
    """Wrapper for llama.cpp Python bindings to provide OpenAI-like interface"""

    # Arguments that should always be preserved
    PRESERVED_ARGS = {'response_model', 'stream', 'max_tokens'}

    def __init__(self, model_path: str, **kwargs):
        """Initialize the LlamaWrapper with a model path

        Args:
            model_path (str): Path to the GGUF model file
            **kwargs: Additional arguments to pass to Llama
        """
        try:
            from llama_cpp import Llama
            import instructor
            self.llm = Llama(model_path=model_path, **kwargs)
            self.chat = self
            self.completions = self
            # Apply instructor patch directly
            instructor.patch(self)
        except ImportError as e:
            raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python") from e
        except Exception as e:
            raise Exception(f"Failed to initialize Llama model: {str(e)}") from e

    @staticmethod
    def custom_instructor_patch(client: 'LlamaWrapper', mode: str = "json") -> 'LlamaWrapper':
        """Custom patch that filters unsupported arguments before applying instructor's patch"""
        original_create = client.create

        @wraps(original_create)
        def filtered_create(*args: Any, **kwargs: Any) -> Any:
            # Filter out unsupported arguments, but preserve essential ones
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if k in client.PRESERVED_ARGS or (k not in client.UNSUPPORTED_ARGS)
            }
            logger.debug(f"Original kwargs: {kwargs}")
            logger.debug(f"Filtered kwargs: {filtered_kwargs}")
            return original_create(*args, **filtered_kwargs)

        # Replace create with filtered version
        client.create = filtered_create
        return instructor.patch(client)

    def create(self, messages=None, prompt=None, stream=False, **kwargs):
        """Create a completion request

        Args:
            messages: List of messages to send to the model
            prompt: Text prompt to send to the model
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to the model

        Returns:
            CompletionResponse or Generator[StreamingResponse]
        """
        # Convert messages to prompt if needed
        if messages and not prompt:
            # Simple concatenation for now
            prompt = messages[-1]['content']

        # Set default max_tokens if not provided
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 2048  # Increased default max_tokens

        # Add temperature and top_p if not provided
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.7
        if 'top_p' not in kwargs:
            kwargs['top_p'] = 0.9

        # Log the final kwargs for debugging
        logger.debug(f"Final create_completion kwargs: {{'prompt': {prompt!r}, 'max_tokens': {kwargs['max_tokens']}, 'stream': {stream}}}")

        if stream:
            logger.debug("Created completion generator")
            return self.StreamingGenerator(self.llm, prompt, **kwargs)

        # Non-streaming response
        try:
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=kwargs.get('max_tokens', 2048),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                stream=False
            )
            return CompletionResponse(chunk=response)
        except Exception as e:
            logger.error(f"Error in create_completion: {str(e)}")
            raise

    class StreamingGenerator(Generator[StreamingResponse, None, None]):
        """Generator for streaming responses"""
        def __init__(self, llm, prompt, **kwargs):
            """Initialize the streaming generator

            Args:
                llm: The llama.cpp model instance
                prompt: The prompt to send to the model
                **kwargs: Additional arguments to pass to create_completion
            """
            self.llm = llm
            self.prompt = prompt
            self.kwargs = kwargs
            self._iterator = None
            self.choices = []  # Add choices attribute for instructor compatibility

        def send(self, value):
            """Send value to generator"""
            raise StopIteration

        def throw(self, typ, val=None, tb=None):
            """Throw exception in generator"""
            raise StopIteration

        def _generate(self):
            """Generate streaming responses"""
            try:
                stream = self.llm.create_completion(
                    prompt=self.prompt,
                    max_tokens=self.kwargs.get('max_tokens', 2048),
                    temperature=self.kwargs.get('temperature', 0.7),
                    top_p=self.kwargs.get('top_p', 0.9),
                    stream=True
                )

                for chunk in stream:
                    if isinstance(chunk, dict):
                        if 'choices' in chunk:
                            # Handle llama-cpp response format
                            choice = chunk['choices'][0]
                            text = choice.get('text', '')
                            finish_reason = choice.get('finish_reason')
                        else:
                            # Handle raw dict format
                            text = chunk.get('text', '')
                            finish_reason = chunk.get('finish_reason')
                    else:
                        text = getattr(chunk, 'text', '')
                        finish_reason = getattr(chunk, 'finish_reason', None)

                    # Skip empty chunks
                    if not text.strip():
                        continue

                    # Update choices for instructor compatibility
                    self.choices = [
                        Choice(
                            index=0,
                            delta={"role": "assistant", "content": text},
                            finish_reason=finish_reason
                        )
                    ]

                    # Create streaming response with the extracted text
                    response = StreamingResponse(
                        choices=[
                            Choice(
                                index=0,
                                delta={"role": "assistant", "content": text},
                                finish_reason=finish_reason
                            )
                        ],
                        id=f"chatcmpl-{hash(str(chunk))& 0xFFFFFFFF:08x}",
                        created=int(datetime.now().timestamp()),
                        model="llama",
                        object_type="chat.completion.chunk"
                    )
                    logger.debug(f"Yielding chunk: {text}")
                    yield response

            except Exception as e:
                logger.error(f"Error in streaming generation: {str(e)}")
                raise

        def __iter__(self):
            """Return self as iterator"""
            return self

        def __next__(self):
            """Get next streaming response"""
            if self._iterator is None:
                self._iterator = self._generate()
            return next(self._iterator)
