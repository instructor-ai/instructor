# --------------------------------------------------------------------------------
# The following code is adapted from a comment on GitHub in the pydantic/pydantic repository by silviumarcu.
# Source: https://github.com/pydantic/pydantic/issues/6381#issuecomment-1831607091
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
# --------------------------------------------------------------------------------

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing import TypeVar, NoReturn, get_args, get_origin, Optional, Generic, List
from copy import deepcopy

Model = TypeVar("Model", bound=BaseModel)

class PartialBase:
    pass

class Partial(Generic[Model]):
    """Generate a new class with all attributes optionals.

    Notes:
        This will wrap a class inheriting form BaseModel and will recursively
        convert all its attributes and its children's attributes to optionals.

    Example:
        Partial[SomeModel]
    """

    def __new__(
        cls,
        *args: object,  # noqa :ARG003
        **kwargs: object,  # noqa :ARG003
    ) -> "Partial[Model]":
        """Cannot instantiate.

        Raises:
            TypeError: Direct instantiation not allowed.
        """
        raise TypeError("Cannot instantiate abstract Partial class.")

    def __init_subclass__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> NoReturn:
        """Cannot subclass.

        Raises:
           TypeError: Subclassing not allowed.
        """
        raise TypeError("Cannot subclass {}.Partial".format(cls.__module__))

    def __class_getitem__(  # type: ignore[override]
        cls,
        wrapped_class: type[Model],
    ) -> type[Model]:
        """Convert model to a partial model with all fields being optionals."""

        def _make_field_optional(
            field: FieldInfo,
        ) -> tuple[object, FieldInfo]:
            tmp_field = deepcopy(field)

            annotation = field.annotation

            # Handle generics (like List, Dict, etc.)
            if get_origin(annotation) is not None:
                # Get the generic base (like List, Dict) and its arguments (like User in List[User])
                generic_base = get_origin(annotation)
                generic_args = get_args(annotation)


                # Recursively apply Partial to each of the generic arguments
                modified_args = tuple(
                    Partial[arg] if isinstance(arg, type) and issubclass(arg, BaseModel) else arg
                    for arg in generic_args
                )

                # Reconstruct the generic type with modified arguments
                tmp_field.annotation = Optional[generic_base[modified_args]]
                tmp_field.default = None
            # If the field is a BaseModel, then recursively convert it's
            # attributes to optionals.
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                tmp_field.annotation = Optional[Partial[annotation]]  # type: ignore[assignment, valid-type]
                tmp_field.default = {}
            else:
                tmp_field.annotation = Optional[field.annotation]  # type: ignore[assignment]
                tmp_field.default = None
            return tmp_field.annotation, tmp_field

        return create_model(  # type: ignore[no-any-return, call-overload]
            f"Partial{wrapped_class.__name__}",
            __base__=(wrapped_class, PartialBase),
            __module__=wrapped_class.__module__,
            **{
                field_name: _make_field_optional(field_info)
                for field_name, field_info in wrapped_class.model_fields.items()
            },
        ) 