"""Build the sole current geometry-only front end (GCC/Clang/MinGW or MSVC)."""

from pathlib import Path
import argparse
import os
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiler")
    a = p.parse_args()
    compiler = a.compiler or shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if not compiler:
        raise SystemExit("A C++17 compiler is required; specify --compiler")
    out = root / "native/build"
    out.mkdir(parents=True, exist_ok=True)
    target = out / ("lazytheta_runner.exe" if os.name == "nt" else "lazytheta_runner")
    source = root / "native/lazytheta_main.cpp"
    if Path(compiler).name.lower() in ("cl", "cl.exe"):
        command = [compiler, "/nologo", "/W4", "/O2", "/std:c++17", "/EHsc", str(source), "/Fe:" + str(target)]
    else:
        command = [compiler, "-Wall", "-Wextra", "-O2", "-std=c++17", str(source), "-o", str(target)]
    subprocess.run(command, check=True, cwd=out)
    print(target)


if __name__ == "__main__":
    main()
