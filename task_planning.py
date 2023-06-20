# Proof of Concept for a task planning and execution system using
# OpenAIs Functions and topological sort, based on the idea in 
# question_answer_subquery.py. 
# This version is simplified and centralizes the execution logic
# in contrast tu the recursive approach in question_answer_subquery.py

import openai
import asyncio
from pydantic import Field
from typing import List, Generator
from openai_function_call import OpenAISchema


class QueryAnswer(OpenAISchema):
    query_id: int
    answer: str


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

    def execute(self, subquery_answers: List[QueryAnswer] | None) -> QueryAnswer:
        return QueryAnswer(query_id=self.id, answer=f"Answer to {self.query}")

    async def aexecute(self, subquery_answers: List[QueryAnswer] | None) -> QueryAnswer:
        return QueryAnswer(query_id=self.id, answer=f"Answer to {self.query}")


class QueryPlan(OpenAISchema):
    """
    Container class representing a tree of queries and subqueries.
    Make sure every task is in the tree, and every task is done only once.
    """

    query_graph: List[Query] = Field(
        ...,
        description="List of queries and subqueries that need to be done to complete the main query. Consists of the main query and its dependencies.",
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


def get_execution_order(query_plan: QueryPlan) -> List[int]:
    """
    Returns the order in which the queries should be executed using topological sort.
    Inspired by https://gitlab.com/ericvsmith/toposort/-/blob/master/src/toposort.py
    """
    tmp_dep_graph = {
        item["id"]: set(dep for dep in item["subqueries"])
        for item in query_plan.dict()["query_graph"]
    }

    def topological_sort(
        dep_graph: dict[int, set[int]]
    ) -> Generator[set[int], None, None]:
        while True:
            ordered = set(item for item, dep in dep_graph.items() if len(dep) == 0)
            if not ordered:
                break
            yield ordered
            dep_graph = {
                item: (dep - ordered)
                for item, dep in dep_graph.items()
                if item not in ordered
            }
        if len(dep_graph) != 0:
            raise ValueError(
                f"Circular dependencies exist among these items: {{{', '.join(f'{key}:{value}' for key, value in dep_graph.items())}}}"
            )

    result = []
    for d in topological_sort(tmp_dep_graph):
        result.extend(sorted(d))
    return result


def execute_queries_in_order(query_plan: QueryPlan) -> dict[int, QueryAnswer]:
    """
    Executes the queries in the query plan in the correct order.
    """
    execution_order = get_execution_order(query_plan)
    query_dict = {q.id: q for q in query_plan.query_graph}
    subquery_answers = {}
    for query_id in execution_order:
        subquery_answers[query_id] = query_dict[query_id].execute(
            subquery_answers=[
                subquery_answers[i] for i in query_dict[query_id].subqueries
            ]
        )
    return subquery_answers


async def aexecute_queries_in_order(query_plan: QueryPlan) -> dict[int, QueryAnswer]:
    """
    Executes the queries in the query plan in the correct order using asyncio and chunks with answered dependencies.
    """
    execution_order = get_execution_order(query_plan)
    query_dict = {q.id: q for q in query_plan.query_graph}
    subquery_answers = {}
    while True:
        ready_to_execute = [
            query_dict[i]
            for i in execution_order
            if i not in subquery_answers
            and all(s in subquery_answers for s in query_dict[i].subqueries)
        ]
        print(ready_to_execute)
        computed_answers = await asyncio.gather(
            *[
                q.aexecute(
                    subquery_answers=[
                        a
                        for a in subquery_answers.values()
                        if a.query_id in q.subqueries
                    ]
                )
                for q in ready_to_execute
            ]
        )
        for answer in computed_answers:
            subquery_answers[answer.query_id] = answer
        if len(subquery_answers) == len(execution_order):
            break
    return subquery_answers


if __name__ == "__main__":
    from pprint import pprint

    plan = query_planner(
        "What is the difference in populations of Canada and #the Jason's home country?"
    )
    answers = execute_queries_in_order(plan)
    # alternativ: asyncio.run(aexecute_queries_in_order(plan)
    pprint(answers)

    """{1: QueryAnswer(query_id=1, answer='Answer to What is the population of Canada?'),
        2: QueryAnswer(query_id=2, answer="Answer to What is Jason's home country?"),
        3: QueryAnswer(query_id=3, answer="Answer to What is the population of Jason's home country?"),
        4: QueryAnswer(query_id=4, answer="Answer to What is the difference in populations of Canada and Jason's home country?")}
    """