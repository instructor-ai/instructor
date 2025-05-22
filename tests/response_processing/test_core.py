"""Tests for core response processing functionality."""

import pytest
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from instructor.dsl.iterable import IterableBase
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import AdapterBase
from instructor.function_calls import OpenAISchema
from instructor.response_processing.core import (
    is_typed_dict,
    prepare_response_model,
    process_response,
    process_response_async,
)


class MockModel(BaseModel):
    """Mock model for testing."""

    name: str
    value: int

    _raw_response: ChatCompletion = None

    @classmethod
    def from_response(cls, response, **kwargs):  # noqa: ARG003  # noqa: ARG003
        """Mock from_response method."""
        return cls(name="test", value=42)


class MockIterableModel(IterableBase):
    """Mock iterable model for testing."""

    tasks: list[str] = ["task1", "task2"]

    @classmethod
    def from_response(cls, response, **kwargs):  # noqa: ARG003  # noqa: ARG003
        """Mock from_response method."""
        return cls()

    @classmethod
    def from_streaming_response(cls, response, **kwargs):  # noqa: ARG003
        """Mock from_streaming_response method."""
        return cls()

    @classmethod
    async def from_streaming_response_async(cls, response, **kwargs):  # noqa: ARG003
        """Mock from_streaming_response_async method."""
        return cls()


class MockAdapterModel(AdapterBase):
    """Mock adapter model for testing."""

    content: str = "adapted content"

    @classmethod
    def from_response(cls, response, **kwargs):  # noqa: ARG003
        """Mock from_response method."""
        return cls()


class MockPartialModel(PartialBase):
    """Mock partial model for testing."""

    @classmethod
    def from_streaming_response(cls, response, **kwargs):  # noqa: ARG003
        """Mock from_streaming_response method."""
        return cls()

    @classmethod
    async def from_streaming_response_async(cls, response, **kwargs):  # noqa: ARG003
        """Mock from_streaming_response_async method."""
        return cls()


class TestProcessResponseAsync:
    """Test async response processing."""

    @pytest.mark.asyncio
    async def test_process_response_async_none_model(self):
        """Test processing with no response model."""
        response = MagicMock(spec=ChatCompletion)
        result = await process_response_async(response, response_model=None)
        assert result is response

    @pytest.mark.asyncio
    async def test_process_response_async_streaming_iterable(self):
        """Test processing streaming iterable model."""
        response = MagicMock(spec=ChatCompletion)

        result = await process_response_async(
            response, response_model=MockIterableModel, stream=True
        )

        assert isinstance(result, MockIterableModel)

    @pytest.mark.asyncio
    async def test_process_response_async_streaming_partial(self):
        """Test processing streaming partial model."""
        response = MagicMock(spec=ChatCompletion)

        result = await process_response_async(
            response, response_model=MockPartialModel, stream=True
        )

        assert isinstance(result, MockPartialModel)

    @pytest.mark.asyncio
    async def test_process_response_async_regular_model(self):
        """Test processing regular model."""
        response = MagicMock(spec=ChatCompletion)

        result = await process_response_async(
            response, response_model=MockModel, stream=False
        )

        assert isinstance(result, MockModel)
        assert result._raw_response is response

    @pytest.mark.asyncio
    async def test_process_response_async_iterable_tasks(self):
        """Test processing iterable model returns tasks."""
        response = MagicMock(spec=ChatCompletion)

        with patch.object(
            MockIterableModel, "from_response", return_value=MockIterableModel()
        ):
            result = await process_response_async(
                response, response_model=MockIterableModel, stream=False
            )

        assert result == ["task1", "task2"]

    @pytest.mark.asyncio
    async def test_process_response_async_adapter_content(self):
        """Test processing adapter model returns content."""
        response = MagicMock(spec=ChatCompletion)

        with patch.object(
            MockAdapterModel, "from_response", return_value=MockAdapterModel()
        ):
            result = await process_response_async(
                response, response_model=MockAdapterModel, stream=False
            )

        assert result == "adapted content"


class TestProcessResponse:
    """Test sync response processing."""

    def test_process_response_none_model(self):
        """Test processing with no response model."""
        response = MagicMock()
        result = process_response(response, response_model=None, stream=False)
        assert result is response

    def test_process_response_streaming_iterable(self):
        """Test processing streaming iterable model."""
        response = MagicMock()

        result = process_response(
            response, response_model=MockIterableModel, stream=True
        )

        assert isinstance(result, MockIterableModel)

    def test_process_response_streaming_partial(self):
        """Test processing streaming partial model."""
        response = MagicMock()

        result = process_response(
            response, response_model=MockPartialModel, stream=True
        )

        assert isinstance(result, MockPartialModel)

    def test_process_response_regular_model(self):
        """Test processing regular model."""
        response = MagicMock()

        result = process_response(response, response_model=MockModel, stream=False)

        assert isinstance(result, MockModel)
        assert result._raw_response is response

    def test_process_response_iterable_tasks(self):
        """Test processing iterable model returns tasks."""
        response = MagicMock()

        with patch.object(
            MockIterableModel, "from_response", return_value=MockIterableModel()
        ):
            result = process_response(
                response, response_model=MockIterableModel, stream=False
            )

        assert result == ["task1", "task2"]

    def test_process_response_adapter_content(self):
        """Test processing adapter model returns content."""
        response = MagicMock()

        with patch.object(
            MockAdapterModel, "from_response", return_value=MockAdapterModel()
        ):
            result = process_response(
                response, response_model=MockAdapterModel, stream=False
            )

        assert result == "adapted content"


class TestIsTypedDict:
    """Test is_typed_dict function."""

    def test_is_typed_dict_true(self):
        """Test identifying a TypedDict."""
        from typing import TypedDict

        class MyTypedDict(TypedDict):
            name: str
            value: int

        assert is_typed_dict(MyTypedDict)

    def test_is_typed_dict_false_regular_dict(self):
        """Test regular dict is not TypedDict."""
        assert not is_typed_dict(dict)

    def test_is_typed_dict_false_other_class(self):
        """Test other classes are not TypedDict."""
        assert not is_typed_dict(str)
        assert not is_typed_dict(MockModel)

    def test_is_typed_dict_false_none(self):
        """Test None is not TypedDict."""
        assert not is_typed_dict(None)


class TestPrepareResponseModel:
    """Test prepare_response_model function."""

    def test_prepare_response_model_none(self):
        """Test preparing None response model."""
        result = prepare_response_model(None)
        assert result is None

    def test_prepare_response_model_simple_type(self):
        """Test preparing simple type."""
        with patch(
            "instructor.response_processing.core.is_simple_type", return_value=True
        ):
            result = prepare_response_model(str)
            # ModelAdapter[str] creates a new class each time, so check the type
            assert result.__name__ == "Response"
            # Check that it's a BaseModel subclass
            assert issubclass(result, BaseModel)

    def test_prepare_response_model_typed_dict(self):
        """Test preparing TypedDict."""
        from typing import TypedDict

        class MyTypedDict(TypedDict):
            name: str
            value: int

        with patch(
            "instructor.response_processing.core.is_typed_dict", return_value=True
        ):
            result = prepare_response_model(MyTypedDict)
            assert issubclass(result, BaseModel)
            assert hasattr(result, "model_fields")

    def test_prepare_response_model_iterable(self):
        """Test preparing Iterable type."""
        from collections.abc import Iterable

        result = prepare_response_model(Iterable[str])
        # IterableModel is a function that returns a class, not a class itself
        assert hasattr(result, "__name__")
        assert hasattr(result, "from_response")

    def test_prepare_response_model_already_openai_schema(self):
        """Test preparing model that's already OpenAISchema."""

        class MySchema(OpenAISchema):
            pass

        result = prepare_response_model(MySchema)
        assert result is MySchema

    def test_prepare_response_model_regular_model(self):
        """Test preparing regular model."""
        with patch(
            "instructor.response_processing.core.openai_schema"
        ) as mock_openai_schema:
            mock_openai_schema.return_value = MockModel
            result = prepare_response_model(MockModel)
            mock_openai_schema.assert_called_once_with(MockModel)
