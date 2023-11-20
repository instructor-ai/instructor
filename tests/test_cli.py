from unittest.mock import patch
from typer.testing import CliRunner
from instructor.cli.cli import app  # replace with the actual import

runner = CliRunner()



def test_jobs_help():
    result = runner.invoke(app, ["jobs", "--help"])
    assert result.exit_code == 0
    assert "Monitor and create fine tuning jobs" in result.stdout

def test_files_help():
    result = runner.invoke(app, ["files", "--help"])
    assert result.exit_code == 0
    assert "Manage files on OpenAI's servers" in result.stdout

def test_usage_help():
    result = runner.invoke(app, ["usage", "--help"])
    assert result.exit_code == 0
    assert "Check OpenAI API usage data" in result.stdout

# Example of 'slow' test
# @pytest.mark.slow
# def test_slow_email_functionality():
#     """This test is slow because it sends an email."""
#     email_client = EmailClient()
#     email_client.send_email()
#     assert email_client.is_email_sent