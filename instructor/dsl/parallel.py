from typing import Type, TypeVar, Union, get_origin, get_args
from instructor.function_calls import OpenAISchema, Mode, openai_schema
from collections.abc import Iterable

T = TypeVar("T")


class ParallelBase:
    def __init__(self, *models: Type[OpenAISchema]):
        # Note that for everything else we've created a class, but for parallel base it is an instance
        assert len(models) > 0, "At least one model is required"
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


def handle_parallel_model(typehint: Type[Iterable[Union[T]]]):
    should_be_iterable = get_origin(typehint)
    assert should_be_iterable is Iterable, f"{should_be_iterable} is not Iterable"

    the_types = get_args(typehint)

    if get_origin(get_args(typehint)[0]) is Union:
        the_types = get_args(get_args(typehint)[0])

    return [
        {"type": "function", "function": openai_schema(model).openai_schema}
        for model in the_types
    ]


def ParallelModel(typehint):
    should_be_iterable = get_origin(typehint)
    assert should_be_iterable is Iterable

    the_types = get_args(typehint)

    if get_origin(get_args(typehint)[0]) is Union:
        the_types = get_args(get_args(typehint)[0])

    return ParallelBase(*[model for model in the_types])
