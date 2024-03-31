import pathlib
import pytest


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize(
    "fpath", pathlib.Path("docs/examples").glob("**/*.md"), ids=str
)
@pytest.mark.skip(reason="This test is not yet implemented")
def test_files_good(fpath):
    from mktestdocs import check_md_file

    check_md_file(fpath=fpath, memory=True)
