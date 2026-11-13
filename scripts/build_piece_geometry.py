"""Build the piece-preserving clipping and certified distance library (GCC/Clang/MinGW or MSVC)."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import numpy as np

root = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler')
    args = parser.parse_args()
    compiler = args.compiler or shutil.which('g++') or shutil.which('clang++') or shutil.which('cl')
    if not compiler:
        raise SystemExit('A C++17 compiler is required; specify --compiler')
    out = root / 'native/build'
    out.mkdir(parents=True, exist_ok=True)
    target = out / ('piece_geometry.dll' if os.name == 'nt' else 'piece_geometry.so')
    source = root / 'native/piece_geometry.cpp'
    raster = root / "native/raster_window.cpp"
    if Path(compiler).name.lower() in ('cl', 'cl.exe'):
        command = [compiler, '/nologo', '/W4', '/O2', '/std:c++17', '/EHsc', '/fp:strict',
                   '/LD', str(source), str(raster),
                   '/link', '/OUT:' + str(target)]
    else:
        command = [compiler, '-Wall', '-Wextra', '-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off',
                   '-shared']
        if os.name != 'nt':
            command.append('-fPIC')
        command.extend([str(source), str(raster), '-o', str(target)])
    subprocess.run(command, check=True, cwd=out)
    print(target)
    # Build with the running interpreter and NumPy headers. The segment engine
    # calls NumPy's native numerical loops, never Python for per-vertex work.
    extension = out / ("_scpsc_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    engine = root / "native/scpsc_native.cpp"
    # NumPy first: Ubuntu's python3-numpy links /usr/include/python3.x/numpy to
    # the system NumPy 1.x headers, which would shadow the running NumPy.
    includes = [np.get_include(), sysconfig.get_path("include")]
    if Path(compiler).name.lower() in ("cl", "cl.exe"):
        command = [compiler, "/nologo", "/W4", "/O2", "/std:c++17", "/EHsc", "/fp:strict", "/LD"]
        command += ["/I" + path for path in includes]
        command += [str(source), str(engine), "/link", "/LIBPATH:" + str(Path(sys.base_prefix) / "libs"),
                    "/OUT:" + str(extension)]
    else:
        command = [compiler, "-Wall", "-Wextra", "-O3", "-std=c++17", "-fno-fast-math",
                   "-ffp-contract=off", "-shared"]
        if os.name != "nt":
            command.append("-fPIC")
        command += ["-I" + path for path in includes]
        command += [str(source), str(engine), "-o", str(extension)]
        if sys.platform == "darwin":
            command += ["-undefined", "dynamic_lookup"]
        elif os.name == "nt":
            command += ["-L" + str(Path(sys.base_prefix) / "libs"),
                        "-lpython" + str(sys.version_info.major) + str(sys.version_info.minor)]
    subprocess.run(command, check=True, cwd=out)
    print(extension)


if __name__ == '__main__':
    main()
