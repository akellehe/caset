# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

from pathlib import Path
import importlib
import os
import shutil
import subprocess
import sys
import tomllib
import platform


TESSERA_ASSERTIONS = os.environ.get("TESSERA_ASSERTIONS")
TESSERA_VERBOSE = os.environ.get("TESSERA_VERBOSE")
TESSERA_ASAN = os.environ.get("TESSERA_ASAN")
TESSERA_CUDA = os.environ.get("TESSERA_CUDA")
LD_PRELOAD = os.environ.get("LD_PRELOAD")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Debug")

def get_scikit_build_dir() -> Path:
    root = Path(__file__).resolve().parent
    pyproj = root / "pyproject.toml"
    data = tomllib.loads(pyproj.read_text())
    template = data["tool"]["scikit-build"]["build-dir"]  # "cmake-build/{wheel_tag}"

    try:
        wheel_tag = subprocess.check_output(
            [sys.executable, "-m", "scikit_build_core.builder.wheel_tag"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "conftest.py's fallback build path needs `scikit-build-core` "
            "to compute the cmake build directory, but it isn't importable "
            "from this Python interpreter ({sys.executable}).\n\n"
            "The usual fix is to install tessera once into the current "
            "environment first:\n\n"
            "    pip install -e \".[dev]\"      # full dev toolchain\n"
            "    pip install -e .              # runtime only (also fine)\n\n"
            "After that, ``pytest tests`` will import the already-built "
            "``tessera._tessera`` extension and skip this fallback "
            "entirely."
            .format(sys=sys)
        ) from exc

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
    if TESSERA_ASAN:
        cmd.append("-DTESSERA_ASAN=ON")
    if TESSERA_VERBOSE:
        cmd.append("-DTESSERA_VERBOSE=ON")
    if TESSERA_ASSERTIONS:
        cmd.append("-DTESSERA_ASSERTIONS=ON")
    if TESSERA_CUDA is not None:
        cmd.append(f"-DTESSERA_CUDA={'ON' if TESSERA_CUDA else 'OFF'}")
    return cmd

def pytest_sessionstart(session):
    # Fast path: tessera._tessera already importable (the usual
    # ``pip install -e .`` outcome).
    try:
        from tessera import _tessera
        if hasattr(_tessera, '__file__') and _tessera.__file__:
            return
    except ImportError:
        pass

    # The remaining branch tries to build tessera from source via cmake
    # so the project author can iterate on C++ without re-running pip.
    # That mode is only viable when scikit-build-core is installed in
    # this Python (it's how we compute the per-Python build directory).
    # If it isn't, the most likely cause is that the user hasn't yet
    # installed tessera into THIS Python interpreter — so guide them
    # there instead of trying to drive a build that won't work anyway.
    try:
        import scikit_build_core  # noqa: F401
    except ImportError:
        venv_hint = ""
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        if not in_venv:
            venv_hint = (
                "\n\nNote: this Python is not running inside a virtualenv. "
                "If you `pip install`-ed into a venv but `pytest` resolved "
                "to a different Python, activate the venv (or run "
                f"``{sys.executable} -m pytest tests`` to force the same "
                "interpreter you installed into).")
        raise RuntimeError(
            "tessera's C extension (``tessera._tessera``) is not "
            f"importable from this Python interpreter:\n"
            f"    {sys.executable}\n\n"
            "Install tessera into THIS Python first:\n\n"
            f"    {sys.executable} -m pip install -e \".[dev]\"   "
            "# full dev toolchain\n"
            f"    {sys.executable} -m pip install -e .          "
            "# runtime only (also fine)\n\n"
            "After that, ``pytest tests`` will pick up the built "
            "extension and skip this fallback path entirely."
            + venv_hint
        )

    build_dir = get_scikit_build_dir()
    env = clean_build_env()
    if not build_dir.exists():
        print("build dir does not exist (", str(build_dir), ") configuring.")
        subprocess.run(get_configure_command(build_dir), check=True, env=env)
    subprocess.run(get_build_command(build_dir), check=True, env=env)

    # cmake --build leaves _tessera*.so at the build-dir root. tessera/__init__.py
    # imports it as ``tessera._tessera``, which requires the .so to live inside
    # the tessera/ package directory that Python resolves first on sys.path —
    # that's the source-tree tessera/ here, since pytest puts the repo root
    # ahead of site-packages. Copy the fresh .so in so the submodule exists.
    pkg_dir = Path(__file__).resolve().parent / "tessera"
    for so_src in build_dir.glob("_tessera*.so"):
        shutil.copy2(so_src, pkg_dir / so_src.name)
    importlib.invalidate_caches()
    sys.modules.pop("tessera", None)
    sys.modules.pop("tessera._tessera", None)

    # If tessera is editable-installed (pip install -e .), the module in
    # site-packages is kept in sync with the build dir by scikit-build-core.
    # Only warn when the module appears to come from a non-editable install.
    spec = importlib.util.find_spec("tessera._tessera")
    if spec is not None and "site-packages" in (spec.origin or ""):
        # Check whether this is an editable install
        try:
            from importlib.metadata import distribution
            dist = distribution("tessera")
            direct_url = dist.read_text("direct_url.json")
            is_editable = direct_url is not None and "editable" in (direct_url or "")
        except Exception:
            is_editable = False
        if not is_editable:
            raise RuntimeError(
                f"Refusing to use tessera._tessera from site-packages: {spec.origin}\n"
                f"Expected to load from: {build_dir}\n"
                "Uninstall via python3 -m pip uninstall tessera before running tests."
            )