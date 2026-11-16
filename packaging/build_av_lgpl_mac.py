r"""build_av_lgpl_mac.py — the macOS (Apple silicon) twin of build_av_lgpl.py: PyAV 17.1.0 on the SAME minimal FFmpeg
8.1.1 under the LGPL-2.1-or-later (PyPI's macOS PyAV wheel bundles libx264 + libx265, GPL). Runs on macOS only (the
GitHub Actions runner of .github/workflows/build-mac.yml); Windows is untouched.

  FFmpeg   the same archive as Windows (GitHub's tarball of tag n8.1.1, FF_SHA of build_av_lgpl.py) and the same
           components (decoders mjpeg / png / tiff / webp / exr, encoders png / exr, the image demuxers, file + pipe,
           the reader's filters); differences: --arch=arm64 --target-os=darwin, no x86 assembly, no static-libgcc
           (clang); zlib = macOS's own /usr/lib/libz (a system library, not bundled)
  toolchain Xcode's clang on the runner (its version is printed into the log); MACOSX_DEPLOYMENT_TARGET 14.0
  PyAV     the av 17.1.0 sdist (AV_SHA of build_av_lgpl.py) built by the TARGET Python (python-build-standalone
           3.10.21 aarch64, the one the app ships) in a venv: Cython 3.3.1, setuptools 79.0.1, wheel 0.48.0; the
           FFmpeg dylibs vendored by delocate 0.13.0
Output: _build/cache/wheels_lgpl_mac/av-17.1.0-cp310-cp310-macosx_14_0_arm64.whl; build_package_mac.py installs it and
checks it. Usage (macOS): python3 packaging/build_av_lgpl_mac.py <target python>
"""
import glob
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_av_lgpl import AV_SHA, AV_URL, FF_SHA, FF_URL, fetch  # noqa: E402  (same pins as Windows)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "_build", "cache")
WORK = os.path.join(REPO, "_build", "ffbuild_mac")          # inside the checkout: no spaces on the runner
PREFIX = os.path.join(WORK, "prefix")
FFMPEG_CONFIGURE_MAC = ["--arch=arm64", "--target-os=darwin", "--cc=clang",
                        "--enable-shared", "--disable-static", "--disable-programs", "--disable-doc", "--disable-debug",
                        "--disable-autodetect", "--disable-everything", "--disable-network", "--enable-zlib",
                        "--enable-decoder=mjpeg,png,tiff,webp,exr", "--enable-encoder=png,exr",
                        "--enable-demuxer=image2,image2pipe,image_*_pipe", "--enable-parser=mjpeg,png,webp",
                        "--enable-protocol=file,pipe",
                        "--enable-filter=buffer,buffersink,pad,fillborders,format,scale,null",
                        "--extra-cflags=-mmacosx-version-min=14.0", "--extra-ldflags=-mmacosx-version-min=14.0"]
TOOLS = ["Cython==3.3.1", "setuptools==79.0.1", "wheel==0.48.0",
         "delocate==0.13.0", "macholib==1.16.3", "packaging==26.3", "typing_extensions==4.16.0"]
OUT = os.path.join(CACHE, "wheels_lgpl_mac")


def run(cmd, **kw):
    print("[av-lgpl-mac]", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("build_av_lgpl_mac.py runs on an Apple silicon Mac (the GitHub Actions macOS runner)")
    target_py = os.path.abspath(sys.argv[1])
    # GIT_CEILING_DIRECTORIES: FFmpeg's version.sh asks `git describe` first; inside the app's checkout it would name
    # the APP's commit. Stopping git at WORK makes it fall back to RELEASE ("8.1.1"), as on Windows (C:\ffbuild).
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="14.0", GIT_CEILING_DIRECTORIES=WORK)
    run(["clang", "--version"])
    # 1 FFmpeg
    tgz = os.path.join(CACHE, "ffmpeg_src", "ffmpeg-n8.1.1.tar.gz")
    fetch(FF_URL, FF_SHA, tgz)
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    run(["tar", "-xzf", tgz, "-C", WORK])
    src = glob.glob(os.path.join(WORK, "FFmpeg-FFmpeg-*"))[0]
    run(["./configure", f"--prefix={PREFIX}"] + FFMPEG_CONFIGURE_MAC, cwd=src, env=env, stdout=subprocess.DEVNULL)
    cfg = open(os.path.join(src, "config.h"), encoding="utf-8", errors="replace").read()
    if 'FFMPEG_LICENSE "LGPL version 2.1 or later"' not in cfg:
        raise SystemExit("FFmpeg's configure did not settle on LGPL-2.1-or-later")
    run(["make", f"-j{os.cpu_count() or 4}"], cwd=src, env=env, stdout=subprocess.DEVNULL)
    ver = open(os.path.join(src, "libavutil", "ffversion.h"), encoding="utf-8", errors="replace").read()
    want = "8.1.1"
    if os.environ.get("SEED_DEFECT") == "ffversion":     # must-fire test run: this check has to stop the build
        want = "0.0.0-seeded"
        print("[av-lgpl-mac] SEEDED DEFECT ffversion: expecting a version FFmpeg is not; the build must stop here",
              flush=True)
    if f'#define FFMPEG_VERSION "{want}"' not in ver:
        raise SystemExit(f"FFmpeg's version string is not {want} (libavutil/ffversion.h):\n{ver}")
    run(["make", "install"], cwd=src, env=env, stdout=subprocess.DEVNULL)
    # 2 PyAV, built by the target Python in its own venv
    sd = os.path.join(CACHE, "pyav_src", "av-17.1.0.tar.gz")
    fetch(AV_URL, AV_SHA, sd)
    x = os.path.join(WORK, "pyav")
    with tarfile.open(sd) as t:
        t.extractall(x, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    venv = os.path.join(WORK, "venv")
    run([target_py, "-m", "venv", venv])
    py = os.path.join(venv, "bin", "python")
    run([py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *TOOLS])
    root = os.path.join(x, "av-17.1.0")
    run([py, "setup.py", "bdist_wheel", f"--ffmpeg-dir={PREFIX}"], cwd=root, env=env)
    whl = glob.glob(os.path.join(root, "dist", "av-17.1.0-*.whl"))[0]
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    run([os.path.join(venv, "bin", "delocate-wheel"), "-v", "-w", OUT, whl],
        env=dict(env, DYLD_LIBRARY_PATH=os.path.join(PREFIX, "lib")))
    out = glob.glob(os.path.join(OUT, "av-17.1.0-*.whl"))
    if len(out) != 1 or "macosx" not in out[0] or "arm64" not in out[0]:
        raise SystemExit(f"expected one macOS arm64 av wheel in {OUT}, found {out}")
    print(f"[av-lgpl-mac] {out[0]} sha256 {hashlib.sha256(open(out[0], 'rb').read()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
