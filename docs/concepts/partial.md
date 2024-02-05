# Streaming Partial Responses

Field level streaming provides incremental snapshots of the current state of the response model that are immediately useable. This approach is particularly relevant in contexts like rendering UI components.

Instructor supports this pattern by making use of `Partial[T]`. This lets us dynamically create a new class that treats all of the original model's fields as `Optional`.

## Understanding Partial Responses

Consider what happens whene we define a response model:

```python
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int
```

If we streamed json out from OpenAI, we would only be able to parse when the object is completed returned!

```
{"name": "Jo
{"name": "John", "ag
{"name": "John", "age":
{"name": "John", "age": 25} # Completed
```

When specifying a `Partial[T]` and setting `stream=True`, the response from `instructor` becomes a `Generator[T]`. As the generator yields results, you can iterate over these incremental updates. The last value yielded by the generator represents the completed extraction!

```
{"name": "Jo                 => User(name="Jo", age=None)
{"name": "John", "ag         => User(name="John", age=None)
{"name": "John", "age:       => User(name="John", age=None)
{"name": "John", "age": 25}  => User(name="John", age=25)
```

!!! warning "Limited Validator Support"

    Fewer validators are supported by `Partial` response models as streamed fields will natural raise validation error, as we do not have a strong opinoin on how to handle them.

Let's look at an example of streaming an extraction of conference information, that would be used to stream in an react component.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from rich.console import Console

client = instructor.patch(OpenAI())

text_block = """
In our recent online meeting, participants from various backgrounds joined to discuss the upcoming tech conference. The names and contact details of the participants were as follows:

- Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
- Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
- Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024, at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher, will be our keynote speaker.

The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities. Each participant is expected to contribute an article to the conference blog by February 20th.

A follow-up meetingis scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
"""


class User(BaseModel):
    name: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    users: List[User]
    date: str
    location: str
    budget: int
    deadline: str


extraction_stream = client.chat.completions.create(
    model="gpt-4",
    response_model=instructor.Partial[MeetingInfo],
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    stream=True,
)


console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    print(obj)
    """
    {'users': None, 'date': None, 'location': None, 'budget': None, 'deadline': None}
    """
    #> {'users': [], 'date': None, 'location': None, 'budget': None, 'deadline': None}
    """
    {
        'users': [{'name': None, 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': '', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': '', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'j', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'joh', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johnd', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@'}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@Tech'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechG'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru44'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'j', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janes', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@Digital',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalD',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alex', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@Code'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster202',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '202',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': '',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena,',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 452',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 500,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '202',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-20',
    }
    """
    #> {'users': [], 'date': None, 'location': None, 'budget': None, 'deadline': None}
    """
    {
        'users': [{'name': None, 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': '', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': '', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'j', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'joh', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johnd', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@'}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@Tech'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechG'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru44'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'j', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janes', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@Digital',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalD',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alex', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@Code'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster202',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '202',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': '',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 452',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 500,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '202',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-20',
    }
    """
    #> {'users': [], 'date': None, 'location': None, 'budget': None, 'deadline': None}
    """
    {
        'users': [{'name': None, 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': '', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': '', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'j', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'joh', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johnd', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@'}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@Tech'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechG'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru44'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'j', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janes', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@Digital',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalD',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alex', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@Code'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster202',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '202',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': '',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena,',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 452',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 500,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '202',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-20',
    }
    """
    #> {'users': [], 'date': None, 'location': None, 'budget': None, 'deadline': None}
    """
    {
        'users': [{'name': None, 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': '', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': '', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'j', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'joh', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johnd', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@'}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@Tech'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechG'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru44'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'j', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janes', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@Digital',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalD',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alex', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@Code'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster202',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '202',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': '',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena,',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 452',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 500,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '202',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-20',
    }
    """
    #> {'users': [], 'date': None, 'location': None, 'budget': None, 'deadline': None}
    """
    {
        'users': [{'name': None, 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': '', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': None, 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': '', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'j', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'joh', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johnd', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': None}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [{'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@'}],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@Tech'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechG'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {'name': 'John Doe', 'email': 'johndoe@email.com', 'twitter': '@TechGuru44'}
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'j', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janes', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {'name': 'Jane Smith', 'email': 'janesmith@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@Digital',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalD',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': None, 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': '', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': None, 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': '', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alex', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': None},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {'name': 'Alex Johnson', 'email': 'alexj@email.com', 'twitter': '@Code'},
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster202',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': None,
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '202',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': None,
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': '',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena,',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 452',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': None,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 500,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': None,
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '202',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-',
    }
    """
    """
    {
        'users': [
            {
                'name': 'John Doe',
                'email': 'johndoe@email.com',
                'twitter': '@TechGuru44',
            },
            {
                'name': 'Jane Smith',
                'email': 'janesmith@email.com',
                'twitter': '@DigitalDiva88',
            },
            {
                'name': 'Alex Johnson',
                'email': 'alexj@email.com',
                'twitter': '@CodeMaster2023',
            },
        ],
        'date': '2024-03-15',
        'location': 'Grand Tech Arena, 4521 Innovation Drive',
        'budget': 50000,
        'deadline': '2024-02-20',
    }
    """
    console.clear()
    console.print(obj)
```

This will output the following:

![Partial Streaming Gif](../img/partial.gif)

## Asynchronous Streaming

I also just want to call out in this example that `instructor` also supports asynchronous streaming. This is useful when you want to stream a response model and process the results as they come in, but you'll need to use the `async for` syntax to iterate over the results.

```python
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

client = instructor.apatch(AsyncOpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


async def print_partial_results():
    user = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_model=instructor.Partial[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    async for m in user:
        print(m)
        #> name=None age=None
        #> name='' age=None
        #> name='Jason' age=None
        #> name='Jason Liu' age=None
        #> name='Jason Liu' age=12


import asyncio

asyncio.run(print_partial_results())
```
