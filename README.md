# caset

Installation requires 

  - Python 3.9 or higher. 

[//]: # (  - `libtorch`, which you can download from [here]&#40;https://pytorch.org/get-started/locally/&#41;, unzip, and place in ./vendor &#40;so the path reads ./vendor/libtorch&#41;.)
  - `cmake`, which you can install via your package manager (e.g. `brew install cmake` on macOS).
  - `pybind11`, which you can install via pip: `pip install pybind11`.
  - `scikit-build-core`, which you can install via pip: `pip install scikit-build-core`.
  - `torch`

You can install all these by running

```bash
python3 -m pip install -r dev-requirements.txt
```

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