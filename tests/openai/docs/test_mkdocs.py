import pathlib
import pytest

from mktestdocs import check_md_file


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize(
    "fpath", pathlib.Path("docs/examples").glob("**/*.md"), ids=str
)
def test_files_good(fpath):
    check_md_file(fpath=fpath, memory=True)
