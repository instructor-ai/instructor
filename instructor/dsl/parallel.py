from typing import Iterable, Type, TypeVar, Union
from instructor.function_calls import OpenAISchema, Mode

T = TypeVar("T")


class ParallelBase:
    def __init__(self, *models: Type[OpenAISchema]):
        # Note that for everything else we've created a class, but for parallel base it is an instance
        self.models = models
        self.registry = {model.__name__: model for model in models}

    def from_response(
        self,
        response,
        mode: Mode,
        validation_context=None,
        strict: bool = None,
    ) -> Iterable[Union[T]]:
        #! We expect this from the OpenAISchema class, We should address
        #! this with a protocol or an abstract class... @jxnlco
        assert mode == Mode.PARALLEL_TOOLS, "Mode must be PARALLEL_TOOLS"
        for tool_call in response.choices[0].message.tool_calls:
            name = tool_call.function.name
            arguments = tool_call.function.arguments
            yield self.registry[name].model_validate_json(
                arguments, context=validation_context, strict=strict
            )
