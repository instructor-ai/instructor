from pydantic import BaseModel, create_model
from typing import Type, TypeVar, Optional, get_type_hints, Generic

T = TypeVar("T", bound=BaseModel)

class PartialBase(BaseModel, Generic[T]):
    """
    A base model for creating Partial models. A Partial model makes all fields of 
    the original model optional, including nested models.
    """

def make_all_fields_optional(model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Recursively make all fields of a Pydantic model optional, including fields of nested models. Renames the model and nested models to "Partial{model.__name__}"
    """
    new_fields = {}
    for name, annotation in get_type_hints(model).items():
        # Check if the field is a nested model
        if issubclass(annotation, BaseModel):
            optional_nested_model = Optional[make_all_fields_optional(annotation)]
            new_fields[name] = (optional_nested_model, None)
        else:
            new_fields[name] = (Optional[annotation], None)
    return create_model(f"Partial{model.__name__}", __base__=PartialBase, **new_fields)


def Partial(model: Type[T]) -> Type[PartialBase[T]]:
    """
    Create a Partial model for a given Pydantic model. This makes all fields (and nested fields) of the model optional.

    ## Usage

    ```python
    from pydantic import BaseModel
    from your_module import Partial

    class Address(BaseModel):
        street: str
        state: str

    class User(BaseModel):
        name: str
        age: str
        address: Address

    PartialUser = Partial(User)
    ```

    ## Result

    ```python
    class PartialUser(BaseModel):
        name: Optional[str] = None
        age: Optional[str] = None
        address: Optional[PartialAddress] = None
    ```

    Parameters:
        model (Type[BaseModel]): The Pydantic model to transform into a Partial model.

    Returns:
        PartialModel (Type[BaseModel]): A new Pydantic model with all fields made optional.
    """
    return make_all_fields_optional(model)
