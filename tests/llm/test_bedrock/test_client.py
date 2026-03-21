from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import Mock, patch

import instructor


def test_from_bedrock_accepts_md_json_mode():
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_client = types.ModuleType("botocore.client")

    class FakeBaseClient:
        pass

    fake_botocore_client.BaseClient = FakeBaseClient

    with patch.dict(
        sys.modules,
        {"botocore": fake_botocore, "botocore.client": fake_botocore_client},
    ):
        bedrock_client = importlib.import_module("instructor.providers.bedrock.client")
        bedrock_client = importlib.reload(bedrock_client)

        client = FakeBaseClient()
        client.converse = Mock()

        instructor_client = bedrock_client.from_bedrock(
            client,
            mode=instructor.Mode.MD_JSON,
        )

        assert instructor_client.mode == instructor.Mode.MD_JSON


def test_from_provider_bedrock_passes_md_json_mode():
    from instructor.auto_client import from_provider

    mock_boto3 = types.ModuleType("boto3")
    mock_boto3.client = Mock(return_value=Mock())

    mock_from_bedrock = Mock(return_value=Mock(mode=instructor.Mode.MD_JSON))

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        with patch.object(
            __import__("instructor"), "from_bedrock", mock_from_bedrock, create=True
        ):
            client = from_provider(
                "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                mode=instructor.Mode.MD_JSON,
            )

    assert client.mode == instructor.Mode.MD_JSON
    mock_from_bedrock.assert_called_once()
    assert mock_from_bedrock.call_args.kwargs["mode"] == instructor.Mode.MD_JSON
