# caset

Installation requires 

  - Python 3.9 or higher. 
  - `libtorch`, which you can download from [here](https://pytorch.org/get-started/locally/), unzip, and place in ./vendor (so the path reads ./vendor/libtorch).
  - `cmake`, which you can install via your package manager (e.g. `brew install cmake` on macOS).
  - `pybind11`, which you can install via pip: `pip install pybind11`.

Build with 

```bash
pip install -e .
```

Build documentation with

```bash
cd docs
make html
```

then open it with 

```bash
open _build/html/index.html
```