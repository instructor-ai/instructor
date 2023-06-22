# Proof of Concept for a task planning and execution system using
# OpenAIs Functions and topological sort, based on the idea in
# question_answer_subquery.py.
# This version is simplified and centralizes the execution logic
# in contrast tu the recursive approach in question_answer_subquery.py
# Added by Jan Philipp Harries / @jpdus

import openai
import asyncio
from pydantic import Field
from typing import List, Generator
from openai_function_call import OpenAISchema
import random


class TaskAnswer(OpenAISchema):
    task_id: int
    result: str


class Task(OpenAISchema):
    """
    Class representing a single task in a task plan.
    """

    id: int = Field(..., description="Unique id of the task")
    task: str = Field(
        ...,
        description="Contains the task in text form. If there are multiple tasks, this task can only be executed when all dependant subtasks have been answered.",
    )
    subtasks: List[int] = Field(
        default_factory=list,
        description="List of the IDs of subtasks that need to be answered before we can answer the main question. Use a subtask when anything may be unknown and we need to ask multiple questions to get the answer. Dependencies must only be other tasks.",
    )

    async def aexecute(self, subtask_answers: List[TaskAnswer] | None) -> TaskAnswer:
        return TaskAnswer(task_id=self.id, answer=f"Answer to {self.task}")


class TaskPlan(OpenAISchema):
    """
    Container class representing a tree of tasks and subtasks.
    Make sure every task is in the tree, and every task is done only once.
    """

    task_graph: List[Task] = Field(
        ...,
        description="List of tasks and subtasks that need to be done to complete the main task. Consists of the main task and its dependencies.",
    )

    async def aexecute_tasks_in_order(self) -> dict[int, TaskAnswer]:
        """
        Executes the tasks in the task plan in the correct order using asyncio and chunks with answered dependencies.
        """
        execution_order = get_execution_order(self)
        task_dict = {q.id: q for q in self.task_graph}
        subtask_answers = {}
        while True:
            ready_to_execute = [
                task_dict[i]
                for i in execution_order
                if i not in subtask_answers
                and all(s in subtask_answers for s in task_dict[i].subtasks)
            ]
            # prints chunks to visualize execution order
            print(ready_to_execute)
            computed_answers = await asyncio.gather(
                *[
                    q.aexecute(
                        subtask_answers=[
                            a
                            for a in subtask_answers.values()
                            if a.task_id in q.subtasks
                        ]
                    )
                    for q in ready_to_execute
                ]
            )
            for answer in computed_answers:
                subtask_answers[answer.task_id] = answer
            if len(subtask_answers) == len(execution_order):
                break
        return subtask_answers


Task.update_forward_refs()
TaskPlan.update_forward_refs()


def task_planner(question: str) -> TaskPlan:
    messages = [
        {
            "role": "system",
            "content": "You are a world class task planning algorithm capable of breaking apart tasks into dependant subtasks, such that the answers can be used to enable the system completing the main task. Do not complete the user task, simply provide a correct compute graph with good specific tasks to ask and relevant subtasks. Before completing the list of tasks, think step by step to get a better understanding the problem.",
        },
        {
            "role": "user",
            "content": f"{question}",
        },
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0,
        functions=[TaskPlan.openai_schema],
        function_call={"name": TaskPlan.openai_schema["name"]},
        messages=messages,
        max_tokens=1000,
    )
    root = TaskPlan.from_response(completion)

    return root


def get_execution_order(task_plan: TaskPlan) -> List[int]:
    """
    Returns the order in which the tasks should be executed using topological sort.
    Inspired by https://gitlab.com/ericvsmith/toposort/-/blob/master/src/toposort.py
    """
    tmp_dep_graph = {
        item["id"]: set(dep for dep in item["subtasks"])
        for item in task_plan.dict()["task_graph"]
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


if __name__ == "__main__":
    from pprint import pprint

    plan = task_planner(
        "What is the difference in populations betweend the adjacent countries of Jan's home country and the adjacent countries of Jason's home country?"
    )

    # randomly shuffle items in the task graph to test execution order
    random.shuffle(plan.task_graph)
    # adjust IDs and subtask IDs
    old_new_id_map = {task.id: i for i, task in enumerate(plan.task_graph)}
    for i, task in enumerate(plan.task_graph):
        task.subtasks = [old_new_id_map[s] for s in task.subtasks]
        task.id = i
    pprint({task.id: task.task for task in plan.task_graph})
    print()
    """
    {0: "Identify the adjacent countries of Jason's home country",
    1: "Calculate the total population of the adjacent countries of Jan's home country",
    2: "Identify the adjacent countries of Jan's home country",
    3: "Identify Jason's home country",
    4: "Get the population of each adjacent country of Jan's home country",
    5: "Get the population of each adjacent country of Jason's home country",
    6: 'Calculate the difference in populations between the adjacent countries of Jan's home country and the adjacent countries of Jason's home country",
    7: "Identify Jan's home country",
    8: "Calculate the total population of the adjacent countries of Jason's home country'}
    """

    task_execution_order = asyncio.run(plan.aexecute_tasks_in_order())

    pprint(task_execution_order, sort_dicts=False)

    """
    done in parallel where possible:

    {3: TaskAnswer(task_id=3, answer="Answer to Identify Jason's home country"),
    7: TaskAnswer(task_id=7, answer="Answer to Identify Jan's home country"),
    0: TaskAnswer(task_id=0, answer="Answer to Identify the adjacent countries of Jason's home country"),
    2: TaskAnswer(task_id=2, answer="Answer to Identify the adjacent countries of Jan's home country"),
    4: TaskAnswer(task_id=4, answer="Answer to Get the population of each adjacent country of Jan's home country"),
    5: TaskAnswer(task_id=5, answer="Answer to Get the population of each adjacent country of Jason's home country"),
    1: TaskAnswer(task_id=1, answer="Answer to Calculate the total population of the adjacent countries of Jan's home country"),
    8: TaskAnswer(task_id=8, answer="Answer to Calculate the total population of the adjacent countries of Jason's home country"),
    6: TaskAnswer(task_id=6, answer="Answer to Calculate the difference in populations between the adjacent countries of Jan's home country and the adjacent countries of Jason's home country")}
    """
