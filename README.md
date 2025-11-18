# caset

# Installation

## Setting up a Python Environment

We recommend using a virtual environment for Python development. `torch` currently has known compatibility issues with 
Python 3.12, so we recommend using Python 3.11. You can create a Python environment pretty easily on a Mac with

```bash
pyenv install 3.11.9
```

Then running your virtual environment command. If you're using virtualenvwrapper that goes like:

```bash
mkvirtualenv caset --python=python3.11
```

## Installing Build Dependencies

To build you'll need to install `scikit-build-core`.

```bash
pip install scikit-build-core
```

Installation requires 

  - Python 3.9 or higher. 
  - A C++17 compatible compiler (e.g. gcc 7+, clang 5+, MSVC 2017+).
  - doxygen (for building documentation).
  - torch (PyTorch) 2.9.0 or higher.
  - scikit-build-core

## Installing caset

To install `caset`, you can use pip. If you're in a virtual environment, make sure it's activated. `cd` to the project 
root then run:

```bash
pip install -v -e .
```

And you'll have an "editable" install of `caset` in your environment. You can test it out by running

```bash
python3 -c "import caset; print(caset.__file__)"
```

## Building Documentation

To build documentation you'll have to install `doxygen` with your package manager; 

On MacOS, use Homebrew:

```bash
brew install doxygen
```

On Ubuntu/Debian, use apt:
```asm
sudo apt-get install doxygen
```

Then install the Python dependencies with pip. It's recommended to use a virtual environment. You can create one with:

```bash
mkvirtualenv caset
```

Then install the dependencies. The development dependencies are listed in `dev-requirements.txt`. Install them using

```bash
python3 -m pip install -r dev-requirements.txt
```

To build the `caset` package there are a couple minor hoops. 

If you run with build isolation (the default), for whatever reason python, pip, or someone fails to keep around the build directory in
/tmp until the build completes, so you'll get weird errors about e.g. torch.h being missing. To work around this you can
Build with 

```bash
python3 -m pip install -v -e . --no-build-isolation
```

Once you've compiled the package; build documentation with

```bash
cd docs && make html
```

then open it with 

```bash
open _build/html/index.html
```