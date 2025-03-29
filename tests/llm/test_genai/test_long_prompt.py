import pytest
import instructor
from pydantic import BaseModel
from .util import models, modes

long_message = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse in augue eu dolor tempus tincidunt. Suspendisse sed magna feugiat, mollis quam at, ornare leo. Praesent lacinia congue risus. Sed ac velit id libero vestibulum posuere. Aenean non lobortis lectus. Donec imperdiet dapibus congue. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Duis blandit dui convallis nisl pellentesque, et eleifend libero tincidunt.

Donec massa orci, finibus eget accumsan vel, hendrerit ac elit. Integer arcu libero, tincidunt in tellus vitae, gravida efficitur risus. Maecenas luctus arcu sed leo eleifend scelerisque. Ut ex dui, ullamcorper vel sapien condimentum, sodales elementum nisl. Nunc pharetra diam lacus, at dapibus turpis dignissim eget. Praesent consectetur sed dolor et pharetra. Quisque rutrum consectetur velit sed lobortis.

Donec a vehicula nulla. Maecenas mattis massa id odio ultrices tincidunt. Nullam tempor, sem id tempus finibus, enim urna gravida elit, sed interdum libero lacus at nisi. Nulla lectus tellus, suscipit et purus a, facilisis eleifend metus. Etiam ac vulputate tortor. Etiam rhoncus lacinia diam in ullamcorper. Cras id arcu justo. Cras interdum, ligula eu eleifend sollicitudin, magna ante tincidunt leo, at iaculis ligula nisi at justo. Suspendisse fringilla sapien ex, sit amet ultrices ligula scelerisque in. Nullam tempus convallis magna. Donec sodales congue felis, vitae cursus odio ultrices vitae.

Donec dapibus eros tortor, ut porta quam elementum sit amet. In quam elit, lobortis viverra hendrerit at, tincidunt quis neque. Maecenas consectetur est a orci iaculis, ut congue nisl gravida. Quisque blandit sapien erat, et ullamcorper elit congue ut. Aenean condimentum porttitor odio, ac eleifend sapien consequat nec. Suspendisse sed eros nec tellus rutrum rutrum consectetur id massa. Vivamus volutpat neque enim, a dignissim sapien venenatis non. Etiam non sapien eu tellus sollicitudin tincidunt non ut tortor. Aliquam semper justo tincidunt mauris tincidunt imperdiet. Donec porttitor felis ac pharetra commodo.

Proin a egestas ligula. Suspendisse ultrices, lacus non accumsan vestibulum, quam metus interdum quam, sed pellentesque mi augue sed libero. Sed sed diam eget felis feugiat accumsan viverra quis magna. Nunc condimentum laoreet mattis. Proin id purus vitae felis aliquet condimentum. Nullam augue lectus, vestibulum sed lacus laoreet, suscipit finibus leo. Donec sed justo sapien. Nullam ac imperdiet nisi. Sed nec convallis ante. Integer volutpat sit amet elit vel iaculis. Nam euismod bibendum dolor nec facilisis. Cras ornare risus sed ex aliquam, eu fringilla metus sollicitudin. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse in augue eu dolor tempus tincidunt. Suspendisse sed magna feugiat, mollis quam at, ornare leo. Praesent lacinia congue risus. Sed ac velit id libero vestibulum posuere. Aenean non lobortis lectus. Donec imperdiet dapibus congue. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Duis blandit dui convallis nisl pellentesque, et eleifend libero tincidunt.

Donec massa orci, finibus eget accumsan vel, hendrerit ac elit. Integer arcu libero, tincidunt in tellus vitae, gravida efficitur risus. Maecenas luctus arcu sed leo eleifend scelerisque. Ut ex dui, ullamcorper vel sapien condimentum, sodales elementum nisl. Nunc pharetra diam lacus, at dapibus turpis dignissim eget. Praesent consectetur sed dolor et pharetra. Quisque rutrum consectetur velit sed lobortis.

Donec a vehicula nulla. Maecenas mattis massa id odio ultrices tincidunt. Nullam tempor, sem id tempus finibus, enim urna gravida elit, sed interdum libero lacus at nisi. Nulla lectus tellus, suscipit et purus a, facilisis eleifend metus. Etiam ac vulputate tortor. Etiam rhoncus lacinia diam in ullamcorper. Cras id arcu justo. Cras interdum, ligula eu eleifend sollicitudin, magna ante tincidunt leo, at iaculis ligula nisi at justo. Suspendisse fringilla sapien ex, sit amet ultrices ligula scelerisque in. Nullam tempus convallis magna. Donec sodales congue felis, vitae cursus odio ultrices vitae.

Donec dapibus eros tortor, ut porta quam elementum sit amet. In quam elit, lobortis viverra hendrerit at, tincidunt quis neque. Maecenas consectetur est a orci iaculis, ut congue nisl gravida. Quisque blandit sapien erat, et ullamcorper elit congue ut. Aenean condimentum porttitor odio, ac eleifend sapien consequat nec. Suspendisse sed eros nec tellus rutrum rutrum consectetur id massa. Vivamus volutpat neque enim, a dignissim sapien venenatis non. Etiam non sapien eu tellus sollicitudin tincidunt non ut tortor. Aliquam semper justo tincidunt mauris tincidunt imperdiet. Donec porttitor felis ac pharetra commodo.

Proin a egestas ligula. Suspendisse ultrices, lacus non accumsan vestibulum, quam metus interdum quam, sed pellentesque mi augue sed libero. Sed sed diam eget felis feugiat accumsan viverra quis magna. Nunc condimentum laoreet mattis. Proin id purus vitae felis aliquet condimentum. Nullam augue lectus, vestibulum sed lacus laoreet, suscipit finibus leo. Donec sed justo sapien. Nullam ac imperdiet nisi. Sed nec convallis ante. Integer volutpat sit amet elit vel iaculis. Nam euismod bibendum dolor nec facilisis. Cras ornare risus sed ex aliquam, eu fringilla metus sollicitudin. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
"""


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_long_prompt(client, model, mode):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": long_message
                + long_message
                + long_message
                + "ivan is 27 years old",
            },
        ],
        response_model=User,
    )
    assert response.name == "ivan"
    assert response.age == 27
