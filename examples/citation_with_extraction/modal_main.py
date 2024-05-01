import modal
from main import app

stub = modal.Stub("rag-citation")

image = modal.Image.debian_slim().pip_install("fastapi", "instructor>=0.2.1", "regex")


@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return app
