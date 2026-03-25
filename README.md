# caset

A C++/pybind11 extension for lattice spacetime and causal set simulations.

## Requirements

- Python 3.9+
- A C++20 compatible compiler (gcc 7+, clang 5+, MSVC 2017+)
- CMake 3.18+ and Ninja (installed automatically via pip)

## Installation

### Setting up a Python environment

We recommend using a virtual environment. You can create one with pyenv and virtualenvwrapper:

```bash
pyenv install 3.13.12
mkvirtualenv caset --python=python3.13
```

### Installing caset

To install in editable (development) mode with all dev dependencies:

```bash
pip install -e ".[dev]"
```

Or for a minimal install without dev tools:

```bash
pip install -e .
```

Verify it works:

```bash
python3 -c "import caset; print(caset.__file__)"
```

## Running Tests

Tests use pytest and are located in `tests/`. The `conftest.py` will automatically build the C++ extension via CMake before running tests.

```bash
pytest tests/ -v
```

### Build options

You can enable additional build features via environment variables:

```bash
CASET_ASAN=1 pytest tests/        # AddressSanitizer + UBSan
CASET_VERBOSE=1 pytest tests/     # C++ level logging
CASET_ASSERTIONS=1 pytest tests/  # Extra assertions for early failure detection
```

## Building Documentation

Install doxygen with your system package manager:

```bash
# macOS
brew install doxygen

# Ubuntu/Debian
sudo apt-get install doxygen
```

Then build the docs:

```bash
cd docs && pip install -r requirements-docs.txt && make html
```

Open the result:

```bash
open _build/html/index.html
```
