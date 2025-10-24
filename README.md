# caset

Installation requires 

  - Python 3.9 or higher. 
  - A C++17 compatible compiler (e.g. gcc 7+, clang 5+, MSVC 2017+).
  - doxygen (for building documentation).
  - torch (PyTorch) 2.9.0 or higher.

You'll have to install `doxygen` with your package manager; 

On MacOS, use Homebrew:
```asm
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
cd docs
make html
```

then open it with 

```bash
open _build/html/index.html
```