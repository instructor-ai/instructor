import instructor
import pytest
from pydantic import BaseModel
import os
from .util import models, modes
import base64


class ImageDescription(BaseModel):
    items: list[str]


curr_file = os.path.dirname(__file__)
image_file = os.path.join(curr_file, "../../assets/image.jpg")
audio_file = os.path.join(curr_file, "../../assets/gettysburg.wav")
audio_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"

pdf_path = os.path.join(curr_file, "../../assets/invoice.pdf")

pdf_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
pdf_base64 = base64.b64encode(open(pdf_path, "rb").read()).decode("utf-8")
pdf_base64_string = f"data:application/pdf;base64,{pdf_base64}"


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
                "content": "Return true if you are provided with an image that you can describe visually in your prompt. This does not include image paths or urls that might point to URLs. ",
            },
            {
                "role": "user",
                "content": [image_file],
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
                "content": "Do you see any blueberries in the context that you've been provided?. ",
            },
            {
                "role": "user",
                "content": [image_file],
            },
        ],
        response_model=bool,
        autodetect_images=autodetect_images,
    )

    assert autodetect_images == response


class Invoice(BaseModel):
    total: float
    items: list[str]


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("pdf_source", [pdf_path, pdf_base64_string, pdf_url])
def test_local_pdf(client, model, mode, pdf_source):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "How much is the invoice?",
                    instructor.multimodal.PDF.autodetect(pdf_source),
                ],
            }
        ],
        response_model=Invoice,
    )
    assert isinstance(response, Invoice)
    assert response.total == 220
    assert len(response.items) == 2


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_existing_genai_file_pdf_integration(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "How much is the invoice?",
                    instructor.multimodal.PDFWithGenaiFile.from_new_genai_file(
                        pdf_path
                    ),
                ],
            }
        ],
        response_model=Invoice,
    )
    assert isinstance(response, Invoice)
    assert response.total == 220
    assert len(response.items) == 2


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_upload_file_genai_pdf_integration(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    file = client.files.upload(file=pdf_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "How much is the invoice?",
                    instructor.multimodal.PDFWithGenaiFile.from_existing_genai_file(
                        file.name
                    ),
                ],
            }
        ],
        response_model=Invoice,
    )
    assert isinstance(response, Invoice)
    assert response.total == 220
    assert len(response.items) == 2


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("pdf_source", [pdf_path, pdf_base64_string, pdf_url])
def test_local_pdf_with_genai_file(client, model, mode, pdf_source):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "How much is the invoice?",
                    instructor.multimodal.PDFWithGenaiFile.autodetect(pdf_source),
                ],
            }
        ],
        response_model=Invoice,
    )
    assert isinstance(response, Invoice)
    assert response.total == 220
    assert len(response.items) == 2
