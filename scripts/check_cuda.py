#!/usr/bin/env python3
"""Diagnose CUDA setup for tessera.

Checks whether CUDA is installed and configured correctly for CMake's
find_package(CUDAToolkit) to work.  Suggests fixes for common problems.

Usage:
    python scripts/check_cuda.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def info(msg):  print(f"    {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")


def find_nvcc_on_path():
    return shutil.which("nvcc")


def get_nvcc_version(nvcc_path):
    try:
        r = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if "release" in line.lower():
                return line.strip()
    except Exception:
        pass
    return None


def check_standard_install():
    """Check /usr/local/cuda (the path CMake expects)."""
    cuda = Path("/usr/local/cuda")
    result = {"exists": False, "is_symlink": False, "target": None,
              "has_nvcc": False, "has_runtime_h": False, "has_cudart": False}
    if cuda.exists():
        result["exists"] = True
        result["is_symlink"] = cuda.is_symlink()
        if result["is_symlink"]:
            result["target"] = str(cuda.resolve())
        result["has_nvcc"] = (cuda / "bin" / "nvcc").exists()
        result["has_runtime_h"] = any(
            (cuda / d / "cuda_runtime.h").exists()
            for d in ["include", "targets/x86_64-linux/include"]
        )
        result["has_cudart"] = any(
            p.exists()
            for d in ["lib64", "targets/x86_64-linux/lib"]
            for p in [(cuda / d / "libcudart.so")]
        )
    return result


def check_hpc_sdk():
    """Check for NVIDIA HPC SDK CUDA installs."""
    results = []
    hpc_base = Path("/opt/nvidia/hpc_sdk")
    if not hpc_base.exists():
        return results
    for arch_dir in sorted(hpc_base.iterdir()):
        if not arch_dir.is_dir():
            continue
        for ver_dir in sorted(arch_dir.iterdir()):
            cuda_dir = ver_dir / "cuda"
            if not cuda_dir.is_dir():
                continue
            for sub in sorted(cuda_dir.iterdir()):
                nvcc = sub / "bin" / "nvcc"
                if nvcc.exists():
                    results.append({
                        "path": str(sub),
                        "nvcc": str(nvcc),
                        "version_dir": sub.name,
                        "has_runtime_h": (sub / "targets/x86_64-linux/include/cuda_runtime.h").exists(),
                        "has_cudart": (sub / "targets/x86_64-linux/lib/libcudart.so").exists(),
                    })
    return results


def check_apt_package():
    """Check for Ubuntu's nvidia-cuda-toolkit package."""
    try:
        r = subprocess.run(["dpkg", "-l", "nvidia-cuda-toolkit"],
                           capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if line.startswith("ii"):
                parts = line.split()
                return {"installed": True, "version": parts[2] if len(parts) > 2 else "unknown"}
    except Exception:
        pass
    return {"installed": False}


def check_cmake_finds_cuda():
    """Try a minimal CMake configure to see if CUDAToolkit is found."""
    import tempfile
    cml = "cmake_minimum_required(VERSION 3.18)\nproject(test LANGUAGES CXX CUDA)\nfind_package(CUDAToolkit REQUIRED)\nmessage(STATUS \"CUDAToolkit_VERSION=${CUDAToolkit_VERSION}\")\nmessage(STATUS \"CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}\")\n"
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "CMakeLists.txt").write_text(cml)
        build = Path(td) / "build"
        r = subprocess.run(
            ["cmake", "-S", td, "-B", str(build), "-G", "Ninja"],
            capture_output=True, text=True, timeout=30
        )
        version = None
        includes = None
        for line in r.stdout.splitlines():
            if "CUDAToolkit_VERSION=" in line:
                version = line.split("=", 1)[1]
            if "CUDAToolkit_INCLUDE_DIRS=" in line:
                includes = line.split("=", 1)[1]
        return {
            "success": r.returncode == 0,
            "version": version,
            "includes": includes,
            "stderr": r.stderr,
        }


def main():
    print(f"{BOLD}tessera CUDA Diagnostic{RESET}")
    print("=" * 50)
    issues = []
    fixes = []

    # 1. nvcc on PATH
    header("1. nvcc on PATH")
    nvcc = find_nvcc_on_path()
    if nvcc:
        ver = get_nvcc_version(nvcc)
        ok(f"Found: {nvcc}")
        if ver:
            info(ver)
    else:
        fail("nvcc not found on PATH")
        issues.append("nvcc not on PATH")
        fixes.append("Add CUDA bin directory to PATH:\n"
                      "    export PATH=/usr/local/cuda/bin:$PATH")

    # 2. Ubuntu nvidia-cuda-toolkit package
    header("2. Ubuntu nvidia-cuda-toolkit package")
    apt = check_apt_package()
    if apt["installed"]:
        warn(f"Installed (version {apt['version']})")
        info("This is Ubuntu's packaged CUDA — it may conflict with")
        info("the HPC SDK or a manual /usr/local/cuda install.")
    else:
        ok("Not installed (no conflict)")

    # 3. NVIDIA HPC SDK
    header("3. NVIDIA HPC SDK")
    hpc = check_hpc_sdk()
    if hpc:
        for h in hpc:
            ok(f"CUDA {h['version_dir']} at {h['path']}")
            if not h["has_runtime_h"]:
                warn("  Missing cuda_runtime.h")
            if not h["has_cudart"]:
                warn("  Missing libcudart.so")
    else:
        info("Not installed")

    # 4. /usr/local/cuda (what CMake expects)
    header("4. /usr/local/cuda")
    std = check_standard_install()
    if std["exists"]:
        if std["is_symlink"]:
            ok(f"Symlink → {std['target']}")
        else:
            ok("Directory exists")
        if std["has_nvcc"]:
            ok("bin/nvcc present")
        else:
            fail("bin/nvcc missing")
            issues.append("/usr/local/cuda/bin/nvcc missing")
        if std["has_runtime_h"]:
            ok("cuda_runtime.h present")
        else:
            fail("cuda_runtime.h missing")
            issues.append("cuda_runtime.h missing from /usr/local/cuda")
        if std["has_cudart"]:
            ok("libcudart.so present")
        else:
            fail("libcudart.so missing")
            issues.append("libcudart.so missing from /usr/local/cuda")
    else:
        fail("/usr/local/cuda does not exist")
        issues.append("/usr/local/cuda not found")
        if hpc:
            best = hpc[-1]  # latest version
            fixes.append(
                f"Create a symlink to your HPC SDK CUDA:\n"
                f"    sudo ln -s {best['path']} /usr/local/cuda"
            )
        else:
            fixes.append(
                "Install the CUDA Toolkit:\n"
                "    See docs/source/fixing_cuda.md for instructions"
            )

    # 5. CMake find_package(CUDAToolkit)
    header("5. CMake find_package(CUDAToolkit)")
    if shutil.which("cmake"):
        cm = check_cmake_finds_cuda()
        if cm["success"]:
            ok(f"CUDAToolkit found (version {cm['version']})")
            if cm["includes"]:
                info(f"Includes: {cm['includes']}")
        else:
            fail("CMake cannot find CUDAToolkit")
            # Check for the specific "non-existent path" error
            if "non-existent path" in cm["stderr"]:
                fail("Stale include path in CUDAToolkit metadata")
                info("This usually means the toolkit layout is non-standard.")
                issues.append("CMake CUDAToolkit has stale include paths")
                if hpc and not std["exists"]:
                    best = hpc[-1]
                    fixes.append(
                        f"Fix with a symlink:\n"
                        f"    sudo ln -s {best['path']} /usr/local/cuda"
                    )
            elif "NOTFOUND" in cm["stderr"] or "not found" in cm["stderr"].lower():
                issues.append("CMake cannot find CUDA compiler or toolkit")
    else:
        warn("cmake not found — cannot test")

    # Summary
    header("Summary")
    if not issues:
        print(f"\n  {GREEN}{BOLD}Everything looks good!{RESET}")
        print("  tessera should build with CUDA support (TESSERA_CUDA=ON).\n")
    else:
        print(f"\n  {RED}{BOLD}Found {len(issues)} issue(s):{RESET}")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

        if fixes:
            print(f"\n  {YELLOW}{BOLD}Suggested fix(es):{RESET}")
            for fix in fixes:
                for line in fix.splitlines():
                    print(f"    {line}")
                print()

        print(f"  For full instructions: docs/source/fixing_cuda.md\n")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
