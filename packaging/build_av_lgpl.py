r"""build_av_lgpl.py — builds the PyAV wheel the app ships: PyAV 17.1.0 (the version ComfyUI uses) on a MINIMAL FFmpeg
8.1.1 under the LGPL-2.1-or-later, so the package carries no GPL code (PyPI's PyAV wheel bundles libx264 + libx265,
GPL-2.0-or-later) and the FFmpeg source to publish is one archive.

  FFmpeg   8.1.1 = git tag n8.1.1 (commit 239f2c733de417201d7ad3b3b8b0d9b63285b2b1, signed by the FFmpeg release
           manager) as GitHub's tarball of the tag, sha256 below. Configure = FFMPEG_CONFIGURE: everything disabled,
           then only what the app uses — decoders mjpeg / png / tiff / webp (+ vp8, its dependency) / exr, encoders png /
           exr, the image demuxers, the file + pipe protocols, the filters the reader's alignment graph needs — no
           external library but zlib, linked statically (zlib licence); libgcc / the MinGW-w64 runtime static.
  toolchain MSYS2 base 20260927 (sha256 below) at C:\msys64, MINGW64: gcc 16.2.0-6, binutils 2.47-4, crt / winpthreads
           14.0.0.r569.g34eeafb7d-1, zlib 1.3.2-3, nasm 3.02-1, make 4.4.1-3 (pacman, as installed 2026-11-02)
  PyAV     the av 17.1.0 sdist from PyPI (sha256 below), built with MSVC (Visual Studio 2022 Build Tools) in an isolated
           env: Cython 3.3.1, setuptools 79.0.1, wheel 0.48.0; the FFmpeg DLLs vendored by delvewheel 1.13.2
Output: _build\cache\wheels_lgpl\av-17.1.0-cp310-cp310-win_amd64.whl; build_package.py installs it and checks it.
Reads + writes are byte-identical to ComfyUI's LoadImage / Save Image (checked on 15 test files).
The FFmpeg source archive + this file's FFMPEG_CONFIGURE are what the release publishes for the LGPL (see NOTICE.txt).
Usage: python packaging\build_av_lgpl.py      (MSYS2 must be at C:\msys64 with the packages above)
"""
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "_build", "cache")
WORK = r"C:\ffbuild"                       # no spaces: FFmpeg's configure does not take them
MSYS_BASH = r"C:\msys64\usr\bin\bash.exe"
MSYS_SFX_SHA = "ad336cccfda47758b5e15cda993fbba421115cb0b126697daef1ee4dfe37209f"   # msys2-base-x86_64-20260927.sfx.exe
FF_TAG, FF_COMMIT = "n8.1.1", "239f2c733de417201d7ad3b3b8b0d9b63285b2b1"
FF_URL = "https://api.github.com/repos/FFmpeg/FFmpeg/tarball/n8.1.1"
FF_SHA = "9ecb4cc05c1bdf996513b162aca48d5d934e000eae29e07763d69f5f4c77fa70"
FFMPEG_CONFIGURE = ("--prefix=/c/ffbuild/prefix --arch=x86_64 --target-os=mingw32 "
                    "--enable-shared --disable-static --disable-programs --disable-doc --disable-debug "
                    "--disable-autodetect --disable-everything --disable-network --enable-zlib --enable-x86asm "
                    "--enable-decoder=mjpeg,png,tiff,webp,exr --enable-encoder=png,exr "
                    "--enable-demuxer=image2,image2pipe,'image_*_pipe' --enable-parser=mjpeg,png,webp "
                    "--enable-protocol=file,pipe --enable-filter=buffer,buffersink,pad,fillborders,format,scale,null "
                    "--extra-ldflags='-static-libgcc -static'")
AV_URL = "https://files.pythonhosted.org/packages/source/a/av/av-17.1.0.tar.gz"
AV_SHA = "7f1e71ff621b66253333926f948e00faae11d855b2442133c65128bca64cdeb3"
TOOLS = ["Cython==3.3.1", "setuptools==79.0.1", "wheel==0.48.0", "delvewheel==1.13.2"]
OUT = os.path.join(CACHE, "wheels_lgpl")


def fetch(url, sha, dst):
    if not os.path.isfile(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    if h != sha:
        raise SystemExit(f"{dst}: sha256 {h} != {sha}")
    print(f"[av-lgpl] {os.path.basename(dst)} sha256 ok")


def msys(cmd):
    subprocess.run([MSYS_BASH, "-lc", cmd], check=True, env=dict(os.environ, MSYSTEM="MINGW64"))


def main():
    if not os.path.isfile(MSYS_BASH):
        raise SystemExit("MSYS2 not found at C:\\msys64 (msys2-base-x86_64-20260927.sfx.exe, sha256 "
                         f"{MSYS_SFX_SHA}; then pacman: make diffutils mingw-w64-x86_64-gcc / -nasm / -pkgconf / -zlib)")
    # 1 FFmpeg
    tgz = os.path.join(WORK, "src", "ffmpeg-n8.1.1.tar.gz")
    fetch(FF_URL, FF_SHA, tgz)
    msys("set -e; cd /c/ffbuild && rm -rf build prefix && mkdir build && tar -xzf src/ffmpeg-n8.1.1.tar.gz -C build"
         f" && cd build/FFmpeg-FFmpeg-* && ./configure {FFMPEG_CONFIGURE} >/dev/null"
         " && grep -q 'FFMPEG_LICENSE \"LGPL version 2.1 or later\"' config.h"
         " && make -j\"$(nproc)\" >/dev/null && make install >/dev/null")
    prefix = os.path.join(WORK, "prefix")
    for lib in glob.glob(os.path.join(prefix, "bin", "*.lib")):           # MSVC import libraries -> lib\ for setup.py
        shutil.copy(lib, os.path.join(prefix, "lib"))
    # 2 PyAV
    sd = os.path.join(CACHE, "pyav_src", "av-17.1.0.tar.gz")
    fetch(AV_URL, AV_SHA, sd)
    src = os.path.join(CACHE, "pyav_src", "x")
    shutil.rmtree(src, ignore_errors=True)
    with tarfile.open(sd) as t:
        t.extractall(src, filter="data")
    env_dir = os.path.join(CACHE, "pyav_build_env")
    py = os.path.join(env_dir, "Scripts", "python.exe")
    if not os.path.isfile(py):
        subprocess.run([sys.executable, "-m", "venv", env_dir], check=True)
    subprocess.run([py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *TOOLS], check=True)
    root = os.path.join(src, "av-17.1.0")
    subprocess.run([py, "setup.py", "bdist_wheel", f"--ffmpeg-dir={prefix}"], cwd=root, check=True)
    whl = glob.glob(os.path.join(root, "dist", "av-17.1.0-*.whl"))[0]
    shutil.rmtree(OUT, ignore_errors=True)
    subprocess.run([py, "-m", "delvewheel", "repair", "--add-path", os.path.join(prefix, "bin"), "-w", OUT, whl],
                   check=True)
    out = glob.glob(os.path.join(OUT, "av-17.1.0-*.whl"))[0]
    print(f"[av-lgpl] {out} sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
