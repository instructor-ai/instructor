import builtins

import pytest


def _block_xai_sdk_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "xai_sdk" or name.startswith("xai_sdk."):
            raise ModuleNotFoundError("No module named 'xai_sdk'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_from_provider_xai_requires_optional_extra(monkeypatch: pytest.MonkeyPatch):
    import instructor
    from instructor.core.exceptions import ConfigurationError

    _block_xai_sdk_imports(monkeypatch)

    with pytest.raises(ConfigurationError) as excinfo:
        instructor.from_provider("xai/grok-3-mini", api_key="test-key")

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "uv pip install" in msg


def test_direct_from_xai_has_clear_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    from instructor.core.exceptions import ConfigurationError
    from instructor.providers.xai import client as xai_client

    # The test suite installs all extras in CI, including `xai-sdk`. Force the
    # "missing optional dependency" behavior so we can validate the error
    # message deterministically.
    monkeypatch.setattr(xai_client, "SyncClient", None)
    monkeypatch.setattr(xai_client, "AsyncClient", None)
    monkeypatch.setattr(xai_client, "xchat", None)

    with pytest.raises(ConfigurationError) as excinfo:
        xai_client.from_xai(object())  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "xai-sdk" in msg
