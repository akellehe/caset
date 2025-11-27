from pathlib import Path
import sys
import subprocess
import tomllib
import sysconfig
import importlib


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

def pytest_sessionstart(session):
    build_dir = get_scikit_build_dir()
    subprocess.run(["cmake", "--build", str(build_dir.parent)], check=True)
    sys.path.insert(0, str(build_dir.parent))

    # refuse to import caset from site-packages by accident:
    spec = importlib.util.find_spec("caset")
    if spec is not None and "site-packages" in (spec.origin or ""):
        raise RuntimeError(
            f"Refusing to use caset from site-packages: {spec.origin}\n"
            f"Expected to load from: {build_dir}"
            "Uninstall via python3 -m pip uninstall caset before running tests."
        )