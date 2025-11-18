from pathlib import Path
import subprocess
import tomllib

def get_build_dir():
    pyproj = Path(__file__).parent / "pyproject.toml"
    data = tomllib.loads(pyproj.read_text())
    base = Path(data["tool"]["scikit-build"]["build-dir"]).parent
    return base

def pytest_sessionstart(session):
    build_dir = get_build_dir()
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)