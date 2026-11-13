"""Build the source-triangle clipping library (GCC/Clang/MinGW or MSVC)."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess

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
    target = out / ('source_clip.dll' if os.name == 'nt' else 'source_clip.so')
    source = root / 'native/source_clip.cpp'
    if Path(compiler).name.lower() in ('cl', 'cl.exe'):
        command = [compiler, '/nologo', '/W4', '/O2', '/std:c++17', '/EHsc', '/fp:strict',
                   '/LD', str(source), '/Fo' + str(out / 'source_clip.obj'),
                   '/link', '/OUT:' + str(target)]
    else:
        command = [compiler, '-Wall', '-Wextra', '-O2', '-std=c++17', '-fno-fast-math', '-ffp-contract=off',
                   '-shared']
        if os.name != 'nt':
            command.append('-fPIC')
        command.extend([str(source), '-o', str(target)])
    subprocess.run(command, check=True, cwd=out)
    print(target)


if __name__ == '__main__':
    main()
