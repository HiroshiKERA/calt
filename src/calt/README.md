# For Developers
Please read this section when developing the library.

## Setting Up the Development Environment
Run the commands below to build the library environment inside a virtual environment:

```bash
# Create a virtual environment
uv venv            

# Activate it
source .venv/bin/activate

# Upgrade the build backend
uv pip install --upgrade build 

# Build the wheel
python -m build

# Install the library itself (adjust version as needed)
uv pip install dist/calt_x-0.1.0-py3-none-any.whl  

# Install development dependencies
uv pip install -e ".[dev]"
```

## Linter + Formatter
```bash
# Lint the codebase
uv run ruff check .

# Reformat the code
uv run ruff format .
```

## Uploading to PyPI
First register on the following sites and generate an API key:  
TestPyPI: <https://test.pypi.org/>  
Production PyPI: <https://pypi.org/>

```bash
# Upload to TestPyPI
uv pip install --upgrade twine

# Verify that the distribution is uploadable
twine check dist/*       

# Upload to TestPyPI
twine upload --repository testpypi dist/* 

# Upload to the production PyPI
twine check dist/*       

twine upload dist/*
```
