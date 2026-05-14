"""
Modern FastAPI + Instructor Integration Example (2026)

Shows best practices for using Instructor with FastAPI:
- Dependency injection for instructor client
- Async operations
- Proper error handling
- Type safety with Pydantic
"""
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
import instructor
from typing import List
import os

app = FastAPI(title="Modern Instructor + FastAPI Example")


# --- Pydantic Models ---

class SearchQuery(BaseModel):
    """A structured search query extracted from natural language."""
    title: str = Field(..., description="What this query is searching for")
    query: str = Field(..., description="Detailed search query optimized for semantic search")
    keywords: List[str] = Field(default_factory=list, description="Key terms to emphasize")


class SearchRequest(BaseModel):
    """Input from the user."""
    text: str = Field(..., description="Natural language search request")


class SearchResponse(BaseModel):
    """Structured response with multiple search queries."""
    queries: List[SearchQuery]


# --- Dependency Injection ---

def get_instructor_client():
    """
    Dependency that provides an instructor client.

    This allows us to:
    - Reuse the same client across requests
    - Easily mock in tests
    - Configure provider in one place
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return instructor.from_provider("openai/gpt-4o-mini", api_key=api_key)


# --- API Endpoints ---

@app.post("/search", response_model=SearchResponse)
async def segment_search(
    request: SearchRequest,
    client=Depends(get_instructor_client)
) -> SearchResponse:
    """
    Segment a natural language search request into structured queries.

    Example:
        Input: "Find restaurants with good sushi and parking near downtown"
        Output: [
            SearchQuery(title="Sushi restaurants", query="high-quality sushi restaurants"),
            SearchQuery(title="Parking availability", query="restaurants with parking downtown")
        ]
    """
    try:
        # Use instructor to extract structured data
        result = client.chat.completions.create(
            response_model=List[SearchQuery],
            messages=[
                {
                    "role": "system",
                    "content": """You are a search query optimizer.
                    Break down complex search requests into multiple specific queries.

                    Guidelines:
                    - Expand abbreviations (SSO → Single Sign On)
                    - Create separate queries for different aspects
                    - Be specific and detailed
                    """
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ],
        )

        return SearchResponse(queries=result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process search request: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "instructor-fastapi"}


# --- Example Usage ---

if __name__ == "__main__":
    import uvicorn

    print("""
    Starting FastAPI server with Instructor integration...

    Try it:
        curl -X POST http://localhost:8000/search \\
             -H "Content-Type: application/json" \\
             -d '{"text": "Find ML engineers with Python and FastAPI experience in Europe"}'

    Docs: http://localhost:8000/docs
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000)
