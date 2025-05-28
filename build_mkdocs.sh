# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize virtual environment and install dependencies
uv sync --extra docs
uv run mkdocs build
