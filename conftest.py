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


CASET_ASAN = os.environ.get("CASET_ASAN")
LD_PRELOAD = os.environ.get("LD_PRELOAD")

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
    return ["cmake", "--build", str(build_dir.parent), "--parallel"]

def get_configure_command(build_dir):
    cmd = ["cmake", "-S", ".", "-B", str(build_dir.parent), "-DCMAKE_BUILD_TYPE=Debug", "-G", "Ninja"]
    if CASET_ASAN:
        cmd.append("-DCASET_ASAN=ON")
    return cmd

def pytest_sessionstart(session):
    build_dir = get_scikit_build_dir()
    env = clean_build_env()
    if not build_dir.parent.exists():
        print("build dir does not exist (", str(build_dir.parent), ") configuring.")
        subprocess.run(get_configure_command(build_dir), check=True, env=env)
    subprocess.run(get_build_command(build_dir), check=True, env=env)
    sys.path.insert(0, str(build_dir.parent))

    # refuse to import caset from site-packages by accident:
    spec = importlib.util.find_spec("caset")
    if spec is not None and "site-packages" in (spec.origin or ""):
        raise RuntimeError(
            f"Refusing to use caset from site-packages: {spec.origin}\n"
            f"Expected to load from: {build_dir}"
            "Uninstall via python3 -m pip uninstall caset before running tests."
        )