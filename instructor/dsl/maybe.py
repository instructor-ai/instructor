from pydantic import BaseModel, Field, create_model
from typing import Type, Optional


class MaybeBase(BaseModel):
    result: Optional[BaseModel]
    error: bool = Field(default=False)
    message: Optional[str]

    def __bool__(self):
        return self.result is not None  # type: ignore


def Maybe(model: Type[BaseModel]) -> MaybeBase:
    """
    Create a Maybe model for a given Pydantic model.

    Parameters:
        model (Type[BaseModel]): The Pydantic model to wrap with Maybe.

    Returns:
        MaybeModel (Type[BaseModel]): A new Pydantic model that includes fields for `result`, `error`, and `message`.
    """

    class MaybeBase(BaseModel):
        def __bool__(self):
            return self.result is not None  # type: ignore

    fields = {
        "result": (
            Optional[model],
            Field(
                default=None,
                description="Correctly extracted result from the model, if any, otherwise None",
            ),
        ),
        "error": (bool, Field(default=False)),
        "message": (
            Optional[str],
            Field(
                default=None,
                description="Error message if no result was found, should be short and concise",
            ),
        ),
    }

    MaybeModel = create_model(f"Maybe{model.__name__}", __base__=MaybeBase, **fields)

    return MaybeModel
