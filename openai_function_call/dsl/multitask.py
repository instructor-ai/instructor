from pydantic import create_model, Field
from typing import Optional, List, Type
from ..function_calls import OpenAISchema


def MultiTask(
    subtask_class: Type[OpenAISchema],
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """
    Dynamically create a MultiTask OpenAISchema that can be used to segment multiple
    tasks given a base class. This creates class that can be used to create a toolkit
    for a specific task, names and descriptions are automatically generated. However
    they can be overridden.

    :param subtask_class: The base class to use for the MultiTask
    :param name: The name of the MultiTask
    :param description: The description of the MultiTask

    :return: new schema class called `Multi{subtask_class.name}`
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

    new_cls = create_model(name, tasks=list_tasks, __base__=(OpenAISchema,))

    new_cls.__doc__ = (
        f"Correct segmentation of `{task_name}` tasks"
        if description is None
        else description
    )

    return new_cls
