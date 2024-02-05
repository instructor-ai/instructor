Pydantic models also support `Union` types, which are used to represent a value that can be one of several types.

While many libraries support multiple function calls, and tool calls support multiple returns, the goal is to provide only one way to do things.

## Unions for Multiple Types

You can use `Union` types to write _agents_ that can dynamically choose actions - by choosing an output class. For example, in a search and lookup function, the LLM can determine whether to execute another search, lookup or other action.

```python
class Search(BaseModel):
    query: str

    def execute(self):
        return ...


class Lookup(BaseModel):
    key: str

    def execute(self):
        return ...


class Action(BaseModel):
    action: Union[Search, Lookup]

    def execute(self):
        return self.action.execute()
```

See 'examples/union/run.py' for a working example.
