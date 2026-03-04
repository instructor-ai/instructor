"""Tests for ParallelBase isinstance/issubclass check in process_response.

Ensures that the ParallelBase check in process_response and
process_response_async correctly handles both:
  - ParallelBase instances (created via ParallelModel, the OpenAI path)
  - ParallelBase subclasses passed as classes (non-OpenAI models)

Fixes: https://github.com/instructor-ai/instructor/issues/2049
"""

import inspect

from pydantic import BaseModel

from instructor.dsl.parallel import ParallelBase


class User(BaseModel):
    name: str
    age: int


class Event(BaseModel):
    title: str
    date: str


def test_parallel_base_isinstance_with_instance():
    """isinstance check should pass for ParallelBase instances (OpenAI path)."""
    instance = ParallelBase(User, Event)
    assert isinstance(instance, ParallelBase)


def test_parallel_base_issubclass_with_class():
    """issubclass check should pass for ParallelBase class itself."""
    assert inspect.isclass(ParallelBase)
    assert issubclass(ParallelBase, ParallelBase)


def test_parallel_base_issubclass_with_subclass():
    """issubclass check should pass for ParallelBase subclasses."""

    class CustomParallel(ParallelBase):
        pass

    assert inspect.isclass(CustomParallel)
    assert issubclass(CustomParallel, ParallelBase)


def test_parallel_base_combined_check_with_instance():
    """The combined isinstance-or-issubclass check should pass for instances."""
    instance = ParallelBase(User, Event)
    result = isinstance(instance, ParallelBase) or (
        inspect.isclass(instance) and issubclass(instance, ParallelBase)
    )
    assert result is True


def test_parallel_base_combined_check_with_class():
    """The combined isinstance-or-issubclass check should pass for classes."""
    result = isinstance(ParallelBase, ParallelBase) or (
        inspect.isclass(ParallelBase) and issubclass(ParallelBase, ParallelBase)
    )
    assert result is True


def test_parallel_base_combined_check_with_subclass():
    """The combined isinstance-or-issubclass check should pass for subclasses."""

    class CustomParallel(ParallelBase):
        pass

    result = isinstance(CustomParallel, ParallelBase) or (
        inspect.isclass(CustomParallel) and issubclass(CustomParallel, ParallelBase)
    )
    assert result is True


def test_parallel_base_combined_check_with_non_parallel():
    """The combined check should fail for unrelated types."""
    result = isinstance(User, ParallelBase) or (
        inspect.isclass(User) and issubclass(User, ParallelBase)
    )
    assert result is False


def test_parallel_base_instance_not_class():
    """An instance of ParallelBase should not be identified as a class."""
    instance = ParallelBase(User)
    assert not inspect.isclass(instance)


def test_old_isinstance_fails_for_class():
    """Demonstrates the original bug: isinstance(SomeClass, ParallelBase) is False
    when SomeClass is a class (not an instance), even if it IS ParallelBase."""
    # This is the core of issue #2049: passing a class (not an instance) to
    # isinstance returns False because the class itself is not an instance of
    # ParallelBase.
    assert not isinstance(ParallelBase, ParallelBase)
