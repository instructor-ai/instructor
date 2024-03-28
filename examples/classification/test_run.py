import pytest

from examples.planning.run import extract_person, extract_people, Person


@pytest.mark.asyncio
async def test_extract_person():
    # Test the extract_person function with a known input
    text = "John is 45 years old"
    expected_person = Person(name="John", age=45)
    person = await extract_person(text)
    assert (
        person == expected_person
    ), "The extracted person does not match the expected person"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names_and_ages, expected_people",
    [
        (
            ["Alice is 30 years old", "Bob is 24 years old"],
            [Person(name="Alice", age=30), Person(name="Bob", age=24)],
        )
    ],
)
async def test_extract_people(names_and_ages, expected_people):
    # Test the extract_people function with a list of known inputs
    people = await extract_people(names_and_ages)
    assert (
        people == expected_people
    ), "The extracted people do not match the expected people"
