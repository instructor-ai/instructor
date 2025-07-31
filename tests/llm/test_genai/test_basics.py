import pytest
import os
import instructor
from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal


class User(BaseModel):
    name: str
    age: int


class RecipeType(Enum):
    DESSERT = 'dessert'
    MAIN = 'main'


class Recipe(BaseModel):
    recipe_type: RecipeType
    ingredients: list[str]


class RecipeLiteral(BaseModel):
    recipe_type: Literal['dessert', 'main']
    ingredients: list[str]


class RecipeWithAggressivePrompting(BaseModel):
    recipe_type: RecipeType = Field(
        ..., 
        description="The type of recipe - must be exactly 'dessert' for sweet dishes or 'main' for main course dishes. Choose the most appropriate category."
    )
    ingredients: list[str] = Field(
        ..., 
        description="A list of ingredients needed for this recipe"
    )


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
)
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "test",
    reason="GOOGLE_API_KEY not set or invalid",
)
async def test_genai_async_from_provider(mode):
    """Test Google GenAI async client using from_provider with different modes"""
    client = instructor.from_provider(
        "google/gemini-2.5-flash", mode=mode, async_client=True
    )

    user = await client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
)
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "test",
    reason="GOOGLE_API_KEY not set or invalid",
)
def test_genai_sync_from_provider(mode):
    """Test Google GenAI sync client using from_provider with different modes"""
    client = instructor.from_provider("google/gemini-2.5-flash", mode=mode)

    user = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
)
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "test",
    reason="GOOGLE_API_KEY not set or invalid",
)
def test_genai_enum_support_issue_1756(mode):
    """Test for issue #1756: Gemini cannot deal with Enums in the JSON schema
    
    This test reproduces the exact scenario reported by DavidNemeskey where
    Gemini fails to handle Enums in JSON schema when using instructor.
    The same code works with OpenAI and direct genai usage.
    """
    client = instructor.from_provider("google/gemini-2.5-flash", mode=mode)

    recipe = client.chat.completions.create(
        response_model=Recipe,
        messages=[
            {
                "role": "user",
                "content": "Write a recipe.",
            },
        ],
    )

    assert isinstance(recipe, Recipe)
    assert isinstance(recipe.recipe_type, RecipeType)
    assert recipe.recipe_type in [RecipeType.DESSERT, RecipeType.MAIN]
    assert isinstance(recipe.ingredients, list)
    assert len(recipe.ingredients) > 0
    assert all(isinstance(ingredient, str) for ingredient in recipe.ingredients)


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
)
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "test",
    reason="GOOGLE_API_KEY not set or invalid",
)
def test_genai_literal_vs_enum_comparison(mode):
    """Compare Literal vs Enum behavior for issue #1756
    
    Tests whether typing.Literal works where Enum fails for Gemini.
    Based on documentation recommendation to use Literal as Enum alternative.
    """
    client = instructor.from_provider("google/gemini-2.5-flash", mode=mode)
    
    prompt_messages = [
        {
            "role": "user",
            "content": "Write a dessert recipe with chocolate and flour.",
        },
    ]
    
    recipe_literal = client.chat.completions.create(
        response_model=RecipeLiteral,
        messages=prompt_messages,
    )
    
    assert isinstance(recipe_literal, RecipeLiteral)
    assert recipe_literal.recipe_type in ['dessert', 'main']
    assert isinstance(recipe_literal.ingredients, list)
    assert len(recipe_literal.ingredients) > 0
    assert all(isinstance(ingredient, str) for ingredient in recipe_literal.ingredients)
    
    with pytest.raises(instructor.exceptions.InstructorRetryException):
        client.chat.completions.create(
            response_model=Recipe,
            messages=prompt_messages,
        )


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
)
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "test",
    reason="GOOGLE_API_KEY not set or invalid",
)
def test_genai_aggressive_enum_prompting(mode):
    """Test if aggressive prompting helps Enum support for issue #1756
    
    Uses detailed field descriptions and specific instructions to see if
    better prompting can make Enum work with Gemini.
    """
    client = instructor.from_provider("google/gemini-2.5-flash", mode=mode)

    recipe = client.chat.completions.create(
        response_model=RecipeWithAggressivePrompting,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful cooking assistant. When classifying recipes, be very precise about the recipe type. Use 'dessert' for sweet dishes and 'main' for main course dishes."
            },
            {
                "role": "user",
                "content": "Create a chocolate cake recipe. Make sure to specify that this is a dessert recipe and include all necessary ingredients.",
            },
        ],
    )

    assert isinstance(recipe, RecipeWithAggressivePrompting)
    assert isinstance(recipe.recipe_type, RecipeType)
    assert recipe.recipe_type in [RecipeType.DESSERT, RecipeType.MAIN]
    assert isinstance(recipe.ingredients, list)
    assert len(recipe.ingredients) > 0
    assert all(isinstance(ingredient, str) for ingredient in recipe.ingredients)
