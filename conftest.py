# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
import importlib
import os
import subprocess
import sys
import tomllib
import platform


CASET_ASSERTIONS = os.environ.get("CASET_ASSERTIONS")
CASET_VERBOSE = os.environ.get("CASET_VERBOSE")
CASET_ASAN = os.environ.get("CASET_ASAN")
CASET_CUDA = os.environ.get("CASET_CUDA")
LD_PRELOAD = os.environ.get("LD_PRELOAD")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Debug")

def get_scikit_build_dir() -> Path:
    root = Path(__file__).resolve().parent
    pyproj = root / "pyproject.toml"
    data = tomllib.loads(pyproj.read_text())
    template = data["tool"]["scikit-build"]["build-dir"]  # "cmake-build/{wheel_tag}"

    wheel_tag = subprocess.check_output(
        [sys.executable, "-m", "scikit_build_core.builder.wheel_tag"],
        text=True,
    ).strip()

    return (root / template.format(wheel_tag=wheel_tag)).resolve()

def clean_build_env():
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("ASAN_OPTIONS", None)
    env.pop("UBSAN_OPTIONS", None)
    return env

def get_build_command(build_dir):
    return ["cmake", "--build", str(build_dir), "--parallel"]

def get_configure_command(build_dir):
    cmd = ["cmake", "-S", ".", "-B", str(build_dir), "-G", "Ninja"]
    if CMAKE_BUILD_TYPE:
        cmd.append(f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}")
    if CASET_ASAN:
        cmd.append("-DCASET_ASAN=ON")
    if CASET_VERBOSE:
        cmd.append("-DCASET_VERBOSE=ON")
    if CASET_ASSERTIONS:
        cmd.append("-DCASET_ASSERTIONS=ON")
    if CASET_CUDA is not None:
        cmd.append(f"-DCASET_CUDA={'ON' if CASET_CUDA else 'OFF'}")
    return cmd

def pytest_sessionstart(session):
    # Skip cmake rebuild if caset._caset (the C extension) is already importable
    # (e.g. from pip install -e .)
    try:
        from caset import _caset
        if hasattr(_caset, '__file__') and _caset.__file__:
            return  # already available, no rebuild needed
    except ImportError:
        pass

    build_dir = get_scikit_build_dir()
    env = clean_build_env()
    if not build_dir.exists():
        print("build dir does not exist (", str(build_dir), ") configuring.")
        subprocess.run(get_configure_command(build_dir), check=True, env=env)
    subprocess.run(get_build_command(build_dir), check=True, env=env)
    # The C extension (_caset.so) is built into the build dir root.
    # Add it so ``from _caset import *`` in caset/__init__.py can find it.
    sys.path.insert(0, str(build_dir))

    # If caset is editable-installed (pip install -e .), the module in
    # site-packages is kept in sync with the build dir by scikit-build-core.
    # Only warn when the module appears to come from a non-editable install.
    spec = importlib.util.find_spec("caset._caset")
    if spec is not None and "site-packages" in (spec.origin or ""):
        # Check whether this is an editable install
        try:
            from importlib.metadata import distribution
            dist = distribution("caset")
            direct_url = dist.read_text("direct_url.json")
            is_editable = direct_url is not None and "editable" in (direct_url or "")
        except Exception:
            is_editable = False
        if not is_editable:
            raise RuntimeError(
                f"Refusing to use caset._caset from site-packages: {spec.origin}\n"
                f"Expected to load from: {build_dir}\n"
                "Uninstall via python3 -m pip uninstall caset before running tests."
            )