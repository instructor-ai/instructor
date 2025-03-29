import instructor
import pytest
from pydantic import BaseModel
import os
from .util import models, modes


class ImageDescription(BaseModel):
    items: list[str]


curr_file = os.path.dirname(__file__)
image_file = os.path.join(curr_file, "../../assets/image.jpg")
audio_file = os.path.join(curr_file, "../../assets/gettysburg.wav")
audio_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"


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
def test_local_file_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "What is shown in this image?",
                    instructor.Image.from_path(image_file),
                    long_message,
                ],  # type: ignore
            }
        ],
        autodetect_images=True,
        response_model=ImageDescription,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_remote_url_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "Describe this image in detail",
                    "gs://generativeai-downloads/images/scones.jpg",
                    long_message,
                ],  # type: ignore
            }
        ],
        autodetect_images=True,
        response_model=ImageDescription,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_gs_url_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    image_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": ["What do you see in this image?", image_url, long_message],
            }  # type: ignore
        ],
        response_model=ImageDescription,
        autodetect_images=True,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_instructor_image(client, model, mode):
    client = instructor.from_genai(client, mode=mode)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": ["Analyze this image", image_file, long_message],
            }  # type: ignore
        ],
        response_model=ImageDescription,
        autodetect_images=True,
    )
    assert isinstance(response, ImageDescription)
    assert len(response.items) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_audio_from_path(client, model, mode):
    client = instructor.from_genai(client, mode=mode)

    class AudioResponse(BaseModel):
        response: str

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "What is this about?",
                    instructor.Audio.from_path(audio_file),
                    long_message,
                ],  # type: ignore
            }
        ],
        response_model=AudioResponse,
    )
    assert isinstance(response, AudioResponse)
    assert len(response.response) > 0


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_audio_from_url(client, model, mode):
    client = instructor.from_genai(client, mode=mode)

    class AudioResponse(BaseModel):
        response: str

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "What is this about?",
                    instructor.Audio.from_url(audio_url),
                    long_message,
                ],  # type: ignore
            }
        ],
        response_model=AudioResponse,
    )
    assert isinstance(response, AudioResponse)
    assert len(response.response) > 0


@pytest.mark.parametrize("autodetect_images", [True, False])
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_autodetect_images_sync(client, model, mode, autodetect_images):
    client = instructor.from_genai(client, mode=mode)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return true if you have an image that you can describe visually in your prompt? This does not include image paths or urls that might point to URLs. ",
            },
            {
                "role": "user",
                "content": long_message + long_message,
            },
            {
                "role": "user",
                "content": ["./image.png"],
            },
        ],
        response_model=bool,
        autodetect_images=autodetect_images,
    )

    assert autodetect_images == response


@pytest.mark.asyncio()
@pytest.mark.parametrize("autodetect_images", [True, False])
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
async def test_autodetect_images_async(client, model, mode, autodetect_images):
    client = instructor.from_genai(client, mode=mode, use_async=True)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return true if you have an image that you can describe visually in your prompt? This does not include image paths or urls that might point to URLs. ",
            },
            {
                "role": "user",
                "content": long_message + long_message,
            },
            {
                "role": "user",
                "content": ["./image.png"],
            },
        ],
        response_model=bool,
        autodetect_images=autodetect_images,
    )

    assert autodetect_images == response
