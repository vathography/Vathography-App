r"""build_package.py — builds the portable Vathography App zip.

  _build\cache\     the python-build-standalone tarball + the wheelhouse (kept: a rebuild downloads nothing)
  _build\Vathography App\   the package folder (rebuilt from scratch every run)
      runtime\      CPython 3.10.21 python-build-standalone 20260901 (= the dev Python, same sha256) + the wheels
      app\vathography_app\
      models\       empty (the models go beside the app)
      Vathography App.bat      (CRLF)
      LICENSE, NOTICE.txt, README.txt, BUILD_INFO.json
  _build\Vathography_App_<version>.zip

Wheels: requirements.in (the direct imports, pinned) resolved under constraints.txt (the dev env's freeze), so every
version = the version the gates ran on; BUILD_INFO.json records each wheel's file name + sha256. PyTorch + OpenCV are
resolved and installed like the rest, then REMOVED (0.1.7 Part T): the first start downloads them from their publishers.
Usage: python packaging\build_package.py [--no-zip]
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "packaging")
BUILD = os.path.join(REPO, "_build")
CACHE = os.path.join(BUILD, "cache")
WHEELS = os.path.join(CACHE, "wheels")
NAME = "Vathography App"
STAGE = os.path.join(BUILD, NAME)
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"
           "cpython-3.10.21%2B20260901-x86_64-pc-windows-msvc-install_only.tar.gz")
PBS_SHA = "b3edb1dacad300c7117ce4d6df97cd2e411afc0d173db43c224f904deb6253b5"   # = the dev Python's tarball
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# the app's own licence: the node's AGPL-3.0-only text + its output permission + a permission for NVIDIA's libraries
APP_LICENSE = os.path.join(PKG, "LICENSE")
AV_LGPL_DIR = os.path.join(CACHE, "wheels_lgpl")             # packaging\build_av_lgpl.py writes it
AV_FFMPEG_VERSION = "FFmpeg version 8.1.1"                    # the minimal LGPL-2.1 build of build_av_lgpl.py
AV_FFMPEG_LICENCE = b"LGPL version 2.1 or later"
FFMPEG_SRC = r"C:\ffbuild\src\ffmpeg-n8.1.1.tar.gz"           # = build_av_lgpl.py's source archive
FFMPEG_SRC_SHA = "9ecb4cc05c1bdf996513b162aca48d5d934e000eae29e07763d69f5f4c77fa70"
GPL_LIBS = ("libx264", "libx265", "libxvid", "libpostproc", "postproc-")
# removed from the runtime: debug symbols, import libraries and C headers (only for compiling extensions)
STRIP_SUFFIX = (".pdb", ".lib")
STRIP_DIRS = (os.path.join("Lib", "site-packages", "torch", "include"), "include", "libs")
# 0.1.7 Part T: PyTorch + OpenCV are NOT in the zip; the first start downloads each official wheel (pinned in the
# manifest's `third_party`) into <APP_HOME>\third_party\<folder>, found through ONE .pth in site-packages
THIRD_PARTY_TOPS = ("torch", "functorch", "torchgen", "cv2")
THIRD_PARTY_DIST = ("torch-", "opencv_python_headless-")
PTH_NAME = "vathography_third_party.pth"
LEFTOVER_PREFIX = ("cudart", "cudnn", "libiomp", "opencv_")
# content markers of code the zip must not carry (Intel IPP / MKL / OpenMP, NVIDIA CUDA / cuDNN): a renamed file is
# still caught; a scan that finds nothing cannot prove absence
PROPRIETARY_MARKERS = (b"ippicv", b"Intel Corporation", b"Intel(R) IPP", b"Intel(R) MKL", b"Intel(R) OpenMP",
                       b"mkl_", b"libiomp", b"NVIDIA Corporation", b"cudart", b"cuDNN")
BINARY_SUFFIX = (".dll", ".pyd", ".exe", ".so")
# the package manager + its helpers never run in the app
PIP_DIRS = ("pip", "setuptools", "_distutils_hack", "pkg_resources")
PIP_DIST = ("pip-", "setuptools-")
# 0.1.7: hf_xet is never used (huggingface_hub imports it only inside its own
# download / upload functions; the app downloads with urllib) and carries ~74 Rust crates incl. AWS-LC: removed
UNUSED_DIRS = ("hf_xet",)
UNUSED_DIST = ("hf_xet-",)
# 0.1.7 rc4 (second reader #4, #6): files the app can never load. Tix (runtime\tcl\tix8.4.3) is not used, and its
# licence points to documents that do not ship; tkinterdnd2 loads ONLY tkdnd\win-x64 on this 64-bit Windows Python with
# Tcl 8.6 (TkinterDnD.py:60-72: PROCESSOR_ARCHITECTURE of the process = AMD64, Tcl major < 9)
UNUSED_RUNTIME_DIRS = (os.path.join("tcl", "tix8.4.3"),)
# 0.1.7: diffusers files under Creative Commons NON-COMMERCIAL licences, never used
# by the app (imported only under TYPE_CHECKING / DIFFUSERS_SLOW_IMPORT); then the build STOPS on any non-commercial
# header left in a .py of the runtime
NONCOMMERCIAL_PATHS = (os.path.join("diffusers", "pipelines", "bria_fibo"),
                       os.path.join("diffusers", "models", "transformers", "transformer_bria_fibo.py"),
                       os.path.join("diffusers", "pipelines", "dit", "pipeline_dit.py"))
NONCOMMERCIAL_RX = re.compile(r"(?i)non-?commercial|CC[- ]BY[- ]NC")
# 0.1.7: diffusers\models\transformers\__init__.py imports the removed
# transformer_bria_fibo UNGUARDED, so with PyTorch installed every diffusers model import crashed (rc6 G2 P2). The builder
# takes that ONE line out and marks the file as changed (Apache-2.0 4(b)); it stops if the line is not there exactly once
TRANSFORMERS_INIT = os.path.join("diffusers", "models", "transformers", "__init__.py")
BRIA_IMPORT_LINE = b"    from .transformer_bria_fibo import BriaFiboTransformer2DModel"
BRIA_CHANGED_NOTE = (b"    # Changed for Vathography App: the import of transformer_bria_fibo was removed here "
                     b"(that file is not included).")
TKDND_KEEP = "win-x64"

BAT = """@echo off
rem Vathography App - photos (or a folder) dropped on this file open in the window.
cd /d "%~dp0app"
start "" "%~dp0runtime\\pythonw.exe" -E -s -m vathography_app gui %*
"""


def say(*a):
    print("[build]", *a, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    say(" ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit(f"command failed ({r.returncode})")


def fetch_python():
    tgz = os.path.join(CACHE, os.path.basename(PBS_URL).replace("%2B", "+"))
    if not (os.path.isfile(tgz) and sha(tgz) == PBS_SHA):
        say("downloading", PBS_URL)
        tmp = tgz + ".part"
        urllib.request.urlretrieve(PBS_URL, tmp)
        if sha(tmp) != PBS_SHA:
            os.remove(tmp)
            raise SystemExit("python-build-standalone: sha256 mismatch")
        os.replace(tmp, tgz)
    return tgz


def main():
    zip_it = "--no-zip" not in sys.argv
    t0 = time.time()
    os.makedirs(WHEELS, exist_ok=True)
    sys.path.insert(0, REPO)
    from vathography_app import __version__
    tgz = fetch_python()
    if os.path.isdir(STAGE):
        say("removing the previous package folder")
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    runtime = os.path.join(STAGE, "runtime")
    with tarfile.open(tgz) as t:
        t.extractall(BUILD + os.sep + "_x", filter="data")
    os.replace(os.path.join(BUILD, "_x", "python"), runtime)
    os.rmdir(os.path.join(BUILD, "_x"))
    py = os.path.join(runtime, "python.exe")
    req, con = os.path.join(PKG, "requirements.in"), os.path.join(PKG, "constraints.txt")
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    pip = [py, "-s", "-m", "pip"]
    # 1 wheelhouse (only binary wheels, resolved by the TARGET runtime's pip: cp310 win_amd64 tags)
    say("resolving + downloading wheels into the cache")
    r = subprocess.run(pip + ["download", "-d", WHEELS, "--only-binary=:all:", "-r", req, "-c", con,
                              "--extra-index-url", TORCH_INDEX], env=env)
    if r.returncode:
        raise SystemExit("pip download failed")
    # 2 install offline from the wheelhouse
    r = subprocess.run(pip + ["install", "--no-index", "--find-links", WHEELS, "--no-warn-script-location",
                              "-r", req, "-c", con], env=env)
    if r.returncode:
        raise SystemExit("pip install failed")
    # 2b PyAV built against an LGPL-only FFmpeg (packaging\build_av_lgpl.py) replaces PyPI's wheel, which bundles the
    # GPL libraries libx264 + libx265; the build stops if the swap did not happen or any GPL library is left
    lgpl = glob.glob(os.path.join(AV_LGPL_DIR, "av-17.1.0-*.whl"))
    if not lgpl:
        raise SystemExit("no LGPL PyAV wheel: run packaging\\build_av_lgpl.py first")
    r = subprocess.run(pip + ["install", "--no-index", "--no-deps", "--force-reinstall", "--no-warn-script-location",
                              lgpl[0]], env=env)
    if r.returncode:
        raise SystemExit("installing the LGPL PyAV wheel failed")
    av_libs = os.path.join(runtime, "Lib", "site-packages", "av.libs")
    gpl = [f for f in os.listdir(av_libs) if f.lower().startswith(GPL_LIBS)]
    avcodec = glob.glob(os.path.join(av_libs, "avcodec-*.dll"))
    blob = open(avcodec[0], "rb").read() if avcodec else b""
    if gpl or AV_FFMPEG_LICENCE not in blob or b"GPL version 2 or later" in blob or AV_FFMPEG_VERSION.encode() not in blob:
        raise SystemExit(f"PyAV's FFmpeg is not the LGPL build ({AV_FFMPEG_VERSION}; GPL libraries: {gpl})")
    # the install-origin records of a LOCAL wheel name this machine's folders: pip's direct_url.json (a file:// URL)
    # and delvewheel's DELVEWHEEL (its command line) -- removed with their RECORD lines; pip freeze then reads av==17.1.0
    dist = glob.glob(os.path.join(runtime, "Lib", "site-packages", "av-17.1.0.dist-info"))[0]
    for f in ("direct_url.json", "DELVEWHEEL"):
        if os.path.isfile(os.path.join(dist, f)):
            os.remove(os.path.join(dist, f))
    rec = os.path.join(dist, "RECORD")
    lines = open(rec, encoding="utf-8").read().splitlines()
    keep = [l for l in lines if not l.split(",")[0].endswith(("/direct_url.json", "/DELVEWHEEL"))]
    with open(rec + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(keep) + "\n")
    os.replace(rec + ".tmp", rec)
    say(f"PyAV: {AV_FFMPEG_VERSION}, {AV_FFMPEG_LICENCE.decode()}, no GPL library")
    freeze = subprocess.run(pip + ["freeze"], env=env, capture_output=True, text=True).stdout.split()
    # 3 strip what never runs
    removed = 0
    for d in STRIP_DIRS:
        p = os.path.join(runtime, d)
        if os.path.isdir(p):
            removed += sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs)
            shutil.rmtree(p)
    for a, ds, fs in os.walk(runtime):
        for f in fs:
            if f.lower().endswith(STRIP_SUFFIX):
                removed += os.path.getsize(os.path.join(a, f))
                os.remove(os.path.join(a, f))
        for d in [d for d in ds if d == "__pycache__"]:
            shutil.rmtree(os.path.join(a, d))
            ds.remove(d)
    # pip's console-script launchers (runtime\Scripts\*.exe) embed the absolute interpreter path of THIS build folder:
    # a path of the build machine, broken on any other one; the app never calls them (the .bat runs pythonw -m)
    scripts = os.path.join(runtime, "Scripts")
    n_exe = 0
    if os.path.isdir(scripts):
        for f in os.listdir(scripts):
            if f.lower().endswith(".exe"):
                removed += os.path.getsize(os.path.join(scripts, f))
                os.remove(os.path.join(scripts, f))
                n_exe += 1
    site = os.path.join(runtime, "Lib", "site-packages")
    # Part T: PyTorch + OpenCV out of the zip (installed here only so pip resolves and freezes the set the gates ran on;
    # their filter at first start = vathography_app\third_party.py). Torch's own dependencies stay.
    n_tp = 0
    for d in os.listdir(site):
        p = os.path.join(site, d)
        if d in THIRD_PARTY_TOPS or (d.startswith(THIRD_PARTY_DIST) and d.endswith(".dist-info")):
            removed += sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs)
            shutil.rmtree(p)
            n_tp += 1
    if n_tp != len(THIRD_PARTY_TOPS) + len(THIRD_PARTY_DIST):
        raise SystemExit(f"expected torch, functorch, torchgen, cv2 + 2 dist-infos in site-packages, removed {n_tp}")
    # step 4: pip, setuptools and their helpers; ensurepip's bundled wheels; pip's empty wheel records
    n_pip = 0
    for d in os.listdir(site):
        p = os.path.join(site, d)
        if d in PIP_DIRS or (d.startswith(PIP_DIST) and d.endswith(".dist-info")):
            shutil.rmtree(p)
            n_pip += 1
        elif d == "distutils-precedence.pth" or (d.endswith(".whl") and os.path.getsize(p) == 0):
            os.remove(p)
            n_pip += 1
    for p in glob.glob(os.path.join(runtime, "Lib", "ensurepip", "_bundled", "*.whl")):
        removed += os.path.getsize(p)
        os.remove(p)
        n_pip += 1
    n_unused = 0                                         # packages the app never loads
    for d in os.listdir(site):
        if d in UNUSED_DIRS or (d.startswith(UNUSED_DIST) and d.endswith(".dist-info")):
            p = os.path.join(site, d)
            removed += sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs)
            shutil.rmtree(p)
            n_unused += 1
    if n_unused != len(UNUSED_DIRS) + len(UNUSED_DIST):
        raise SystemExit(f"expected {UNUSED_DIRS} + their dist-info in site-packages, removed {n_unused}")
    for d in UNUSED_RUNTIME_DIRS:
        p = os.path.join(runtime, d)
        if not os.path.isdir(p):
            raise SystemExit(f"{d} not found in the runtime -- check the Python build")
        removed += sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs)
        shutil.rmtree(p)
    tkdnd = os.path.join(site, "tkinterdnd2", "tkdnd")
    others = [d for d in os.listdir(tkdnd) if d != TKDND_KEEP]
    if TKDND_KEEP not in os.listdir(tkdnd):
        raise SystemExit(f"tkdnd\\{TKDND_KEEP} not found -- check the tkinterdnd2 version")
    for d in others:
        p = os.path.join(tkdnd, d)
        removed += sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs)
        shutil.rmtree(p)
    if sorted(os.listdir(tkdnd)) != [TKDND_KEEP] or any(os.path.isdir(os.path.join(runtime, d)) for d in UNUSED_RUNTIME_DIRS):
        raise SystemExit("tkdnd platforms or Tix left in the runtime")
    for rel in NONCOMMERCIAL_PATHS:
        p = os.path.join(site, rel)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)
        else:
            raise SystemExit(f"{rel} not found -- check the diffusers version")
    ti = os.path.join(site, TRANSFORMERS_INIT)
    src = open(ti, "rb").read()
    eol = b"\r\n" if b"\r\n" in src else b"\n"
    lines = src.split(eol)
    if lines.count(BRIA_IMPORT_LINE) != 1:
        raise SystemExit(f"{TRANSFORMERS_INIT}: the transformer_bria_fibo import line is not there exactly once "
                         "-- check the diffusers version")
    lines[lines.index(BRIA_IMPORT_LINE)] = BRIA_CHANGED_NOTE
    tmp = ti + ".tmp"
    with open(tmp, "wb") as f:
        f.write(eol.join(lines))
    os.replace(tmp, ti)
    left_bria = [os.path.relpath(os.path.join(a, f), site) for a, _, fs in os.walk(os.path.join(site, "diffusers"))
                 for f in fs if f.endswith(".py") and re.search(rb"^\s*from \.+[\w.]*transformer_bria_fibo import",
                                                                open(os.path.join(a, f), "rb").read(), re.M)]
    if left_bria:
        raise SystemExit("an import of the removed transformer_bria_fibo is left:\n  " + "\n  ".join(left_bria))
    nc = [os.path.relpath(os.path.join(a, f), runtime) for a, _, fs in os.walk(runtime) for f in fs if f.endswith(".py")
          and NONCOMMERCIAL_RX.search(open(os.path.join(a, f), "rb").read(3000).decode("utf-8", "replace"))]
    if nc:
        raise SystemExit("non-commercial licence header left in the runtime:\n  " + "\n  ".join(nc[:20]))
    # ONE .pth, two relative lines: site-packages -> ..\..\.. = the package folder; Python skips a missing folder
    # silently, so the app checks third_party.ready() itself before any work
    man = json.load(open(os.path.join(REPO, "vathography_app", "model_manifest.json"), encoding="utf-8"))
    tp = man["third_party"]
    with open(os.path.join(site, PTH_NAME), "w", encoding="ascii", newline="\r\n") as f:
        f.write("".join(f"..\\..\\..\\third_party\\{e['folder']}\n" for e in tp))
    # STOP on any leftover: by name, then by content (T7's markers) over every binary of the package folder
    left = []
    for a, ds, fs in os.walk(runtime):
        for n in ds + fs:
            low = n.lower()
            # package names only at site-packages' top level (scipy carries a compat folder named "torch");
            # library file prefixes anywhere
            top_pkg = os.path.normcase(a) == os.path.normcase(site) and (low in THIRD_PARTY_TOPS or low in PIP_DIRS
                                                                          or low in UNUSED_DIRS)
            if low.startswith(LEFTOVER_PREFIX) or top_pkg:
                left.append(os.path.join(a, n))
            elif low.endswith(".pth") and n not in (PTH_NAME,):
                left.append(os.path.join(a, n))
            elif low.endswith(".whl") or (low.startswith(PIP_DIST + UNUSED_DIST) and low.endswith(".dist-info")):
                left.append(os.path.join(a, n))
    if left:
        raise SystemExit("left in the runtime, must not ship:\n  " + "\n  ".join(left[:20]))
    hits = []
    for a, _, fs in os.walk(runtime):
        for n in fs:
            if n.lower().endswith(BINARY_SUFFIX):
                blob = open(os.path.join(a, n), "rb").read()
                found = [m.decode() for m in PROPRIETARY_MARKERS if m in blob]
                if found:
                    hits.append(f"{os.path.relpath(os.path.join(a, n), STAGE)}: {', '.join(found)}")
    if hits:
        raise SystemExit("proprietary marker in a binary of the package:\n  " + "\n  ".join(hits[:20]))
    notice = open(os.path.join(PKG, "NOTICE.txt"), encoding="utf-8").read()
    named = [w for w in ("pip ", "pip,", "setuptools") if w in notice]
    if named:
        raise SystemExit(f"NOTICE.txt still names {named}: they are not in the package any more")
    say(f"stripped {removed / 1e9:.2f} GB (debug symbols, import libraries, headers, {n_exe} pip launchers, "
        f"PyTorch + OpenCV ({n_tp} folders, downloaded at first start), pip + setuptools ({n_pip} items)); "
        f"no leftover by name, no proprietary marker in a binary")
    # 4 the app
    shutil.copytree(os.path.join(REPO, "vathography_app"), os.path.join(STAGE, "app", "vathography_app"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp"))
    os.makedirs(os.path.join(STAGE, "models"))
    with open(os.path.join(STAGE, f"{NAME}.bat"), "w", encoding="ascii", newline="\r\n") as f:
        f.write(BAT)
    shutil.copy(APP_LICENSE, os.path.join(STAGE, "LICENSE"))
    for f in ("NOTICE.txt", "README.txt"):
        shutil.copy(os.path.join(PKG, f), os.path.join(STAGE, f))
    # the licence texts NOTICE.txt points to (each verbatim from its source at the pinned version)
    shutil.copytree(os.path.join(PKG, "licences"), os.path.join(STAGE, "licences"))
    # the LGPL FFmpeg's Corresponding Source travels WITH the binary: the tag's archive + its licence text + the build
    # configuration (source\README.txt)
    src_dir = os.path.join(STAGE, "source")
    os.makedirs(src_dir)
    if sha(FFMPEG_SRC) != FFMPEG_SRC_SHA:
        raise SystemExit(f"{FFMPEG_SRC}: not the FFmpeg n8.1.1 archive the DLLs were built from")
    shutil.copy(FFMPEG_SRC, os.path.join(src_dir, "ffmpeg-n8.1.1.tar.gz"))
    with tarfile.open(FFMPEG_SRC) as t:
        m = next(x for x in t.getmembers() if x.name.endswith("/COPYING.LGPLv2.1"))
        with open(os.path.join(src_dir, "COPYING.LGPLv2.1"), "wb") as f:
            f.write(t.extractfile(m).read())
    shutil.copy(os.path.join(PKG, "SOURCE_README.txt"), os.path.join(src_dir, "README.txt"))
    used = sorted(w for w in os.listdir(WHEELS) if w.endswith(".whl"))
    installed = {l.split("==")[0].lower().replace("_", "-") for l in freeze if "==" in l}
    wheels = [dict(file=w, sha256=sha(os.path.join(WHEELS, w))) for w in used
              if w.split("-")[0].lower().replace("_", "-") in installed and w.split("-")[0].lower() != "av"]
    wheels.append(dict(file=os.path.basename(lgpl[0]), sha256=sha(lgpl[0]),
                       built_from="PyAV 17.1.0 sdist + FFmpeg 8.1.1 (tag n8.1.1), LGPL-2.1-or-later, minimal "
                                  "configure (packaging/build_av_lgpl.py)"))
    # freeze + wheels = the record of what the gates ran on (torch + OpenCV included: they are NOT in this package);
    # third_party = what the first start downloads
    info = dict(app=NAME, version=__version__, built=time.strftime("%Y-%m-%d %H:%M"),
                python=dict(source=PBS_URL, sha256=PBS_SHA),
                note="freeze and wheels record the set the gates ran on. Not in this package: PyTorch and OpenCV "
                     "(third_party: downloaded from their publishers at first start), pip and setuptools (never used "
                     "by the app), hf-xet (never used by the app)",
                freeze=freeze, wheels=wheels,
                third_party=[{k: e[k] for k in ("name", "version", "url", "size", "sha256", "folder")} for e in tp])
    with open(os.path.join(STAGE, "BUILD_INFO.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    size = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(STAGE) for f in fs)
    say(f"package folder: {size / 1e9:.2f} GB, {len(freeze)} packages")
    # 5 the zip (crash-safe: .part then replace)
    if zip_it:
        z = os.path.join(BUILD, f"{NAME.replace(' ', '_')}_{__version__}.zip")
        tmp = z + ".part"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
            for a, ds, fs in os.walk(STAGE):
                rel = os.path.relpath(a, BUILD)
                if not fs and not ds:
                    zf.write(a, rel + "/")
                for f in fs:
                    zf.write(os.path.join(a, f), os.path.join(rel, f))
        os.replace(tmp, z)
        say(f"zip: {z} {os.path.getsize(z) / 1e9:.2f} GB sha256 {sha(z)}")
    say(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
