"""
Unit tests for ListResponse class (#1303, #1305)
"""

import pytest
from pydantic import BaseModel
from instructor.dsl import ListResponse


class User(BaseModel):
    """Test user model"""
    name: str
    age: int


class TestListResponseBasics:
    """Test basic ListResponse functionality"""

    def test_listresponse_creation_empty(self):
        """Test creating an empty ListResponse"""
        response = ListResponse()
        assert len(response) == 0
        assert list(response) == []

    def test_listresponse_creation_with_items(self):
        """Test creating ListResponse with items"""
        users = [User(name="John", age=30), User(name="Jane", age=25)]
        response = ListResponse(users)
        assert len(response) == 2
        assert response[0].name == "John"
        assert response[1].name == "Jane"

    def test_listresponse_from_list_factory(self):
        """Test ListResponse.from_list() factory method"""
        users = [User(name="Alice", age=28)]
        raw_response = {"model": "gpt-4", "usage": {"total_tokens": 100}}

        response = ListResponse.from_list(users, raw_response=raw_response)

        assert len(response) == 1
        assert response[0].name == "Alice"
        assert response._raw_response == raw_response

    def test_listresponse_raw_response_storage(self):
        """Test that _raw_response is properly stored and retrieved"""
        mock_response = {"status": "success", "tokens": 42}
        response = ListResponse(_raw_response=mock_response)

        assert response._raw_response == mock_response
        assert response.get_raw_response() == mock_response

    def test_listresponse_raw_response_none(self):
        """Test ListResponse with no raw response"""
        response = ListResponse()
        assert response._raw_response is None
        assert response.get_raw_response() is None


class TestListResponseListOperations:
    """Test that ListResponse works like a normal list"""

    def test_indexing(self):
        """Test list indexing"""
        users = [User(name="A", age=1), User(name="B", age=2)]
        response = ListResponse(users, _raw_response={"data": "test"})

        assert response[0].name == "A"
        assert response[1].name == "B"
        assert response[-1].name == "B"

    def test_slicing(self):
        """Test list slicing"""
        users = [User(name="A", age=1), User(name="B", age=2), User(name="C", age=3)]
        response = ListResponse(users, _raw_response={"data": "test"})

        sliced = response[1:]
        assert len(sliced) == 2
        assert sliced[0].name == "B"

    def test_iteration(self):
        """Test iterating over ListResponse"""
        users = [User(name="A", age=1), User(name="B", age=2)]
        response = ListResponse(users, _raw_response={"data": "test"})

        names = [u.name for u in response]
        assert names == ["A", "B"]

    def test_append(self):
        """Test appending to ListResponse"""
        response = ListResponse(_raw_response={"data": "test"})
        response.append(User(name="John", age=30))

        assert len(response) == 1
        assert response[0].name == "John"
        assert response._raw_response == {"data": "test"}  # Raw response preserved

    def test_extend(self):
        """Test extending ListResponse"""
        response = ListResponse([User(name="A", age=1)], _raw_response={"data": "test"})
        response.extend([User(name="B", age=2), User(name="C", age=3)])

        assert len(response) == 3
        assert response[1].name == "B"
        assert response._raw_response == {"data": "test"}  # Raw response preserved

    def test_pop(self):
        """Test popping from ListResponse"""
        users = [User(name="A", age=1), User(name="B", age=2)]
        response = ListResponse(users, _raw_response={"data": "test"})

        popped = response.pop()
        assert popped.name == "B"
        assert len(response) == 1
        assert response._raw_response == {"data": "test"}  # Raw response preserved

    def test_len(self):
        """Test len() on ListResponse"""
        response = ListResponse([User(name="A", age=1), User(name="B", age=2)])
        assert len(response) == 2

    def test_in_operator(self):
        """Test 'in' operator"""
        user = User(name="John", age=30)
        response = ListResponse([user])

        assert user in response

    def test_enumerate(self):
        """Test enumerate()"""
        users = [User(name="A", age=1), User(name="B", age=2)]
        response = ListResponse(users)

        enumerated = list(enumerate(response))
        assert enumerated[0] == (0, users[0])
        assert enumerated[1] == (1, users[1])

    def test_list_conversion(self):
        """Test converting ListResponse to list"""
        users = [User(name="A", age=1)]
        response = ListResponse(users)

        as_list = list(response)
        assert as_list == users
        assert isinstance(as_list, list)
        assert not isinstance(as_list, ListResponse)


class TestListResponseRepresentation:
    """Test string representations"""

    def test_repr(self):
        """Test repr() without raw response"""
        response = ListResponse([User(name="John", age=30)])
        repr_str = repr(response)
        assert "ListResponse" in repr_str
        assert "John" in repr_str

    def test_repr_with_raw_response(self):
        """Test repr() with raw response"""
        response = ListResponse([User(name="John", age=30)], _raw_response={"data": "test"})
        repr_str = repr(response)
        assert "ListResponse" in repr_str
        assert "_raw_response=..." in repr_str

    def test_str(self):
        """Test str()"""
        response = ListResponse([User(name="John", age=30)])
        str_repr = str(response)
        assert "ListResponse" in str_repr


class TestListResponseRawResponsePreservation:
    """Test that operations preserve raw response"""

    def test_append_preserves_raw_response(self):
        """Verify append preserves _raw_response"""
        raw = {"tokens": 100}
        response = ListResponse(_raw_response=raw)
        response.append(User(name="John", age=30))
        assert response._raw_response is raw

    def test_extend_preserves_raw_response(self):
        """Verify extend preserves _raw_response"""
        raw = {"tokens": 100}
        response = ListResponse(_raw_response=raw)
        response.extend([User(name="John", age=30)])
        assert response._raw_response is raw

    def test_pop_preserves_raw_response(self):
        """Verify pop preserves _raw_response"""
        raw = {"tokens": 100}
        response = ListResponse([User(name="John", age=30)], _raw_response=raw)
        response.pop()
        assert response._raw_response is raw

    def test_clear_preserves_raw_response(self):
        """Verify clear preserves _raw_response"""
        raw = {"tokens": 100}
        response = ListResponse([User(name="John", age=30)], _raw_response=raw)
        response.clear()
        assert response._raw_response is raw
        assert len(response) == 0


class TestListResponseTypeCompatibility:
    """Test type compatibility"""

    def test_is_instance_of_list(self):
        """ListResponse should be instance of list"""
        response = ListResponse()
        assert isinstance(response, list)

    def test_subclass_of_list(self):
        """ListResponse should be subclass of list"""
        assert issubclass(ListResponse, list)

    def test_isinstance_listresponse(self):
        """Should be instance of ListResponse"""
        response = ListResponse()
        assert isinstance(response, ListResponse)
