import pydantic


class User(pydantic.BaseModel):
    name: str
    age: int


from pprint import pprint

pprint(
    {
        "name": User.model_json_schema()["title"],
        "description": User.__doc__,
        "input_schema": User.model_json_schema(),
    }
)
