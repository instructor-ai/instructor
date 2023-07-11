# MultiTask 

Defining a task and creating a list of classes is a common enough pattern that we define a helper function `MultiTask` that dynamically creates a new schema that has a task attribute defined as a list of the task subclass, including some prebuilt prompts and allows us to avoid writing some extra code.

!!! example "Extending user details"

    Using the previous example with extracting `UserDetails` we might want to extract multiple users rather than a single user, `MultiTask` makes it easy!

    ```python
    class UserDetails(OpenAISchema):
        """Details of a user"""
        name: str = Field(..., description="users's full name")
        age: int
    
    MultiUserDetails = MultiTask(UserDetails)
    ```

::: openai_function_call.dsl.multitask