import importlib
import sys


def test_top_level_exports_load_lazily():
    sys.modules.pop("instructor", None)
    sys.modules.pop("instructor.processing.response", None)
    sys.modules.pop("instructor.processing.multimodal", None)

    instructor = importlib.import_module("instructor")

    assert "instructor.processing.response" not in sys.modules
    assert "instructor.processing.multimodal" not in sys.modules
    assert instructor.Mode.TOOLS.value == "tool_call"
    assert "instructor.processing.response" not in sys.modules


def test_top_level_openai_factory_uses_v2_module():
    sys.modules.pop("instructor", None)
    sys.modules.pop("instructor.v2.providers.openai.client", None)

    instructor = importlib.import_module("instructor")

    assert "instructor.v2.providers.openai.client" not in sys.modules
    assert instructor.from_openai is not None
    assert "instructor.v2.providers.openai.client" in sys.modules
