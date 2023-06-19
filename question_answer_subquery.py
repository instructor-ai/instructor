import openai
import enum
import json

from pydantic import Field
from typing import List, Tuple
from openai_function_call import OpenAISchema
from tenacity import retry, stop_after_attempt


class Query(OpenAISchema):
    """
    Class representing a single query in a query plan.
    """

    id: int = Field(..., description="Unique id of the query")
    query: str = Field(
        ...,
        description="Contains the query in text form. If there are multiple queries, this query can only be answered when all dependant subqueries have been answered.",
    )
    subqueries: List[int] = Field(
        default_factory=list,
        description="List of the IDs of subqueries that need to be answered before we can answer the main question. Use a subquery when anything may be unknown and we need to ask multiple questions to get the answer. Dependencies must only be other queries.",
    )

class QueryPlan(OpenAISchema):
    """
    Container class representing a tree of queries and subqueries.
    Make sure every task is in the tree, and every task is done only once.
    """

    query_graph: List[Query] = Field(
        ..., description="List of queries and subqueries that need to be done to complete the main query. Consists of the main query and its dependencies."
    )


Query.update_forward_refs()
QueryPlan.update_forward_refs()


def query_planner(question: str) -> QueryPlan:

    messages = [
        {
            "role": "system",
            "content": "You are a world class query planning algorithm capable of breaking apart queries into dependant subqueries, such that the answers can be used to enable the system completing the main query. Do not complete the user query, simply provide a correct compute graph with good specific queries to ask and relevant subqueries. Before completing the list of queries, think step by step to get a better understanding the problem.",
        },
        {
            "role": "user",
            "content": f"{question}",
        },
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0,
        functions=[QueryPlan.openai_schema],
        function_call={"name": QueryPlan.openai_schema["name"]},
        messages=messages,
        max_tokens=1000,
    )
    root = QueryPlan.from_response(completion)
    return root


if __name__ == "__main__":
    from pprint import pprint

    plan = query_planner(
        "What is the difference in populations of Canada and the Jason's home country?"
    )
    pprint(plan.dict())
    """
    > {'query_graph': [{'id': 1,
                      'query': 'What is the population of Canada?',
                      'subqueries': []},
                     {'id': 2,
                      'query': "What is Jason's home country?",
                      'subqueries': []},
                     {'id': 3,
                      'query': "What is the population of Jason's home country?",
                      'subqueries': [2]},
                     {'id': 4,
                      'query': 'What is the difference in populations of Canada '
                               "and the Jason's home country?",
                      'subqueries': [1, 3]}]}
                      
    plan = query_planner(
    "Write a pitch presentation to promote my startup to VC investors!"
    )
    pprint(plan.dict())
    
    > {'query_graph': [{'id': 1,
                  'query': 'What is the name of the startup?',
                  'subqueries': []},
                 {'id': 2,
                  'query': 'What is the mission of the startup?',
                  'subqueries': []},
                 {'id': 3,
                  'query': 'What problem does the startup solve?',
                  'subqueries': []},
                 {'id': 4,
                  'query': 'What is the unique selling proposition (USP) of '
                           'the startup?',
                  'subqueries': []},
                 {'id': 5,
                  'query': 'Who are the target customers of the startup?',
                  'subqueries': []},
                 {'id': 6,
                  'query': "What is the market size for the startup's product "
                           'or service?',
                  'subqueries': []},
                 {'id': 7,
                  'query': 'Who are the competitors of the startup and how '
                           'does the startup differentiate itself?',
                  'subqueries': []},
                 {'id': 8,
                  'query': 'What is the business model of the startup?',
                  'subqueries': []},
                 {'id': 9,
                  'query': 'What is the current financial status of the '
                           'startup?',
                  'subqueries': []},
                 {'id': 10,
                  'query': 'What is the growth plan of the startup?',
                  'subqueries': []},
                 {'id': 11,
                  'query': 'What is the funding requirement and how will the '
                           'funds be used?',
                  'subqueries': []},
                 {'id': 12,
                  'query': 'Who are the team members and what are their '
                           'backgrounds?',
                  'subqueries': []},
                 {'id': 13,
                  'query': 'Write a pitch presentation to promote the startup '
                           'to VC investors',
                  'subqueries': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}]}
    """
