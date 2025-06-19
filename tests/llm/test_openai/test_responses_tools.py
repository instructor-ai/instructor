import instructor
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import pytest


def test_web_search(client: OpenAI):
    from openai.types.responses import ResponseFunctionWebSearch

    instructor_client = instructor.from_openai(
        client, mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
    )

    class Citation(BaseModel):
        id: int
        url: str

    class Summary(BaseModel):
        citations: list[Citation]
        summary: str

    resp, completion = instructor_client.responses.create_with_completion(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "You must call the web_search tool to answer the user's question.",
            },
            {
                "role": "user",
                "content": "What are some of the best places to visit in New York for someone who likes to eat Latin American Food?",
            },
        ],
        tools=[{"type": "web_search_preview"}],
        response_model=Summary,
    )

    # Validate that a web search was performed
    n_web_searches = 0
    for tool_call in completion.output:
        if isinstance(tool_call, ResponseFunctionWebSearch):
            n_web_searches += 1

    assert n_web_searches >= 1


@pytest.mark.asyncio
async def test_web_search_async(aclient: AsyncOpenAI):
    from openai.types.responses import ResponseFunctionWebSearch

    instructor_client = instructor.from_openai(
        aclient,
        mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    )

    class Citation(BaseModel):
        id: int
        url: str

    class Summary(BaseModel):
        citations: list[Citation]
        summary: str

    resp, completion = await instructor_client.responses.create_with_completion(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "You must call the web_search tool to answer the user's question.",
            },
            {
                "role": "user",
                "content": "What are some of the best places to visit in New York for someone who likes to eat Latin American Food?",
            },
        ],
        tools=[{"type": "web_search_preview"}],
        max_retries=2,
        response_model=Summary,
    )

    # Validate that a web search was performed
    n_web_searches = 0
    for tool_call in completion.output:
        if isinstance(tool_call, ResponseFunctionWebSearch):
            n_web_searches += 1

    assert n_web_searches >= 1


def test_file_search():
    from openai.types.responses import ResponseFileSearchToolCall

    VECTOR_STORE_ID = "vs_682025c8b9f881919fb6fac9cb7b4941"
    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        async_client=False,
    )

    class Citation(BaseModel):
        file_id: int
        file_name: str
        excerpt: str

    class Response(BaseModel):
        citations: list[Citation]
        response: str

    response, completion = client.responses.create_with_completion(
        input="You must use the file_search tool to answer the user's question. How much does the Kyoto itineary cost? Generate a final response as a summary of the information you've found. Provide the exact file_name that you used to generate your response + a short excerpt that shows where you got your answer from",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": 2,
            },
        ],
        response_model=Response,
        max_retries=2,
        include=["file_search_call.results"],
    )

    # Assert that the File Search Tool was used
    file_search_call = 0
    for output_tool in completion.output:
        if isinstance(output_tool, ResponseFileSearchToolCall):
            file_search_call += 1

    assert file_search_call >= 1


@pytest.mark.asyncio
async def test_file_search_async():
    from openai.types.responses import ResponseFileSearchToolCall

    VECTOR_STORE_ID = "vs_682025c8b9f881919fb6fac9cb7b4941"
    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        async_client=True,
    )

    class Citation(BaseModel):
        file_id: int
        file_name: str
        excerpt: str

    class Response(BaseModel):
        citations: list[Citation]
        response: str

    response, completion = await client.responses.create_with_completion(
        input="You must use the file_search tool to answer the user's question. How much does the Kyoto itineary cost? Generate a final response as a summary of the information you've found. Provide the exact file_name that you used to generate your response + a short excerpt that shows where you got your answer from",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": 2,
            },
        ],
        response_model=Response,
        max_retries=2,
        include=["file_search_call.results"],
    )

    # Assert that the File Search Tool was used
    file_search_call = 0
    for output_tool in completion.output:
        if isinstance(output_tool, ResponseFileSearchToolCall):
            file_search_call += 1

    assert file_search_call >= 1
