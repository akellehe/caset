from pathlib import Path
import tomllib

def get_build_dir():
    pyproj = Path(__file__).parent / "pyproject.toml"
    data = tomllib.loads(pyproj.read_text())
    pattern = data["tool"]["scikit-build"]["build-dir"]

    # Resolve {wheel_tag} by locating the actual directory
    base = Path("cmake-build")
    matches = list(base.glob("*"))  # one subdir
    if not matches:
        raise RuntimeError("No scikit-build build directory found.")
    return matches[0]

def pytest_sessionstart(session):
    build_dir = get_build_dir()
    import subprocess
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)