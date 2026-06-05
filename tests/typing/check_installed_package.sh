#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"

uv build --default-index https://pypi.org/simple --wheel \
    --out-dir "$tmp_dir/dist" "$repo_root"

wheels=("$tmp_dir"/dist/*.whl)
uv venv "$tmp_dir/.venv"
uv pip install --default-index https://pypi.org/simple \
    --python "$tmp_dir/.venv/bin/python" \
    "${wheels[0]}" "ty==0.0.44" typing_extensions

VIRTUAL_ENV="$tmp_dir/.venv" "$tmp_dir/.venv/bin/python" -c \
    'from importlib.util import find_spec; from pathlib import Path; spec = find_spec("instructor"); assert spec is not None and spec.submodule_search_locations is not None; assert (Path(next(iter(spec.submodule_search_locations))) / "py.typed").is_file()'

cp "$repo_root/tests/typing/test_installed_package.py" "$tmp_dir/test_installed_package.py"

cd "$tmp_dir"
VIRTUAL_ENV="$tmp_dir/.venv" "$tmp_dir/.venv/bin/ty" check \
    --error-on-warning test_installed_package.py
