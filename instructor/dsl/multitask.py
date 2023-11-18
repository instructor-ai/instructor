from typing import List, Optional, Type

from pydantic import BaseModel, Field, create_model

from instructor.function_calls import OpenAISchema


class MultiTaskBase:
    task_type = None  # type: ignore

    @classmethod
    def from_streaming_response(cls, completion):
        json_chunks = cls.extract_json(completion)
        yield from cls.tasks_from_chunks(json_chunks)

    @classmethod
    def tasks_from_chunks(cls, json_chunks):
        started = False
        potential_object = ""
        for chunk in json_chunks:
            potential_object += chunk
            if not started:
                if "[" in chunk:
                    started = True
                    potential_object = chunk[chunk.find("[") + 1 :]
                continue

            task_json, potential_object = cls.get_object(potential_object, 0)
            if task_json:
                obj = cls.task_type.model_validate_json(task_json)  # type: ignore
                yield obj

    @staticmethod
    def extract_json(completion):
        for chunk in completion:
            if chunk["choices"]:
                delta = chunk["choices"][0]["delta"]
                if "function_call" in delta:
                    if "arguments" in delta["function_call"]:
                        yield delta["function_call"]["arguments"]

    @staticmethod
    def get_object(str, stack):
        for i, c in enumerate(str):
            if c == "{":
                stack += 1
            if c == "}":
                stack -= 1
                if stack == 0:
                    return str[: i + 1], str[i + 2 :]
        return None, str


def MultiTask(
    subtask_class: Type[BaseModel],
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """
    Dynamically create a MultiTask OpenAISchema that can be used to segment multiple
    tasks given a base class. This creates class that can be used to create a toolkit
    for a specific task, names and descriptions are automatically generated. However
    they can be overridden.

    ## Usage

    ```python
    from pydantic import BaseModel, Field
    from instructor import MultiTask

    class User(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")
        role: str = Field(description="The role of the person")

    MultiUser = MultiTask(User)
    ```

    ## Result

    ```python
    class MultiUser(OpenAISchema, MultiTaskBase):
        tasks: List[User] = Field(
            default_factory=list,
            repr=False,
            description="Correctly segmented list of `User` tasks",
        )

        @classmethod
        def from_streaming_response(cls, completion) -> Generator[User]:
            '''
            Parse the streaming response from OpenAI and yield a `User` object
            for each task in the response
            '''
            json_chunks = cls.extract_json(completion)
            yield from cls.tasks_from_chunks(json_chunks)
    ```

    Parameters:
        subtask_class (Type[OpenAISchema]): The base class to use for the MultiTask
        name (Optional[str]): The name of the MultiTask class, if None then the name
            of the subtask class is used as `Multi{subtask_class.__name__}`
        description (Optional[str]): The description of the MultiTask class, if None
            then the description is set to `Correct segmentation of `{subtask_class.__name__}` tasks`

    Returns:
        schema (OpenAISchema): A new class that can be used to segment multiple tasks
    """
    task_name = subtask_class.__name__ if name is None else name

    name = f"Multi{task_name}"

    list_tasks = (
        List[subtask_class],
        Field(
            default_factory=list,
            repr=False,
            description=f"Correctly segmented list of `{task_name}` tasks",
        ),
    )

    new_cls = create_model(
        name,
        tasks=list_tasks,
        __base__=(OpenAISchema, MultiTaskBase),  # type: ignore
    )
    # set the class constructor BaseModel
    new_cls.task_type = subtask_class

    new_cls.__doc__ = (
        f"Correct segmentation of `{task_name}` tasks"
        if description is None
        else description
    )

    return new_cls
