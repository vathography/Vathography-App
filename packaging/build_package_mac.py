r"""build_package_mac.py — builds the macOS (Apple silicon) Vathography App zip. Runs on a Mac only: the GitHub
Actions runner of .github/workflows/build-mac.yml. The Windows build (build_package.py) is untouched;
this file reuses its constants and repeats its steps for the macOS layout.

  _build/cache/                 python-build-standalone (aarch64) + the macOS wheelhouse + the LGPL PyAV wheel
  _build/mac/Vathography App/   the folder the zip holds:
      Vathography App.app/Contents/
          Info.plist            (packaging/macos/Info.plist, version filled in)
          MacOS/Vathography App (packaging/macos/launcher.c, compiled here)
          Resources/runtime/    CPython 3.10.21 python-build-standalone 20260901 aarch64 + the wheels
          Resources/app/vathography_app/   (precompiled: the app never writes into its signed bundle)
          Resources/NOTICE.txt, LICENSE   (the window's "Licences and notices" reads Resources/NOTICE.txt)
          Resources/vathography_app.icns
      README.txt, LICENSE, NOTICE.txt, licences/, source/   (beside the .app, as in the Windows zip)
  _build/Vathography_App_<version>_macos.zip  (or ..._macos_TEST.zip while packaging/macos/README.txt + NOTICE.txt
                                               do not exist: the Windows texts would be wrong on a Mac)

Same rules as Windows: requirements.in + constraints.txt (the CUDA suffix "+cu128" taken off, nothing else changed);
PyTorch + OpenCV installed so pip resolves the gated set, then REMOVED (Q4: downloaded at first start from PyPI, the
manifest's third_party_macos; the wheels downloaded here must be exactly those pins); PyPI's PyAV replaced by the LGPL
build (packaging/build_av_lgpl_mac.py); pip / setuptools / hf_xet / the non-commercial diffusers files removed; the
build STOPS on any leftover by name or by content. Then: every Mach-O signed ad hoc (Q2: no Apple account), the bundle
signed and verified (codesign --verify --deep --strict), the zip written by ditto without extended attributes, and
the zip unpacked again and verified.
Usage (macOS): python3 packaging/build_package_mac.py
"""
import glob
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_package as W  # noqa: E402  (constants + helpers of the Windows build; its main() is not run)

REPO, PKG, BUILD, CACHE, NAME = W.REPO, W.PKG, W.BUILD, W.CACHE, W.NAME
MAC = os.path.join(PKG, "macos")
WHEELS = os.path.join(CACHE, "wheels_mac")
AV_LGPL_DIR = os.path.join(CACHE, "wheels_lgpl_mac")
OUTDIR = os.path.join(BUILD, "mac")
FOLDER = os.path.join(OUTDIR, NAME)
APP = os.path.join(FOLDER, NAME + ".app")
CONTENTS = os.path.join(APP, "Contents")
RES = os.path.join(CONTENTS, "Resources")
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"
           "cpython-3.10.21%2B20260901-aarch64-apple-darwin-install_only.tar.gz")
PBS_SHA = "cee232aabfb6790eec78f3cca935caeb7bd4eedca4dcb0a10dbcdb4302320b38"   # release SHA256SUMS, 2026-11-15
FFMPEG_SRC = os.path.join(CACHE, "ffmpeg_src", "ffmpeg-n8.1.1.tar.gz")        # build_av_lgpl_mac.py fetched it
# SciPy's macOS wheel ships libquadmath (LGPL-2.1-or-later) as a separate dylib: its source goes into source/ (ruled
# W1, 2026-11-16). Its build paths name GCC 13.3.0 for aarch64-apple-darwin23 = Homebrew's gcc@13 13.3.0 (2024-05-22),
# GNU's tarball + Homebrew's Apple silicon patch; the build stops if the shipped library no longer carries that mark.
GCC_URL = "https://ftp.gnu.org/gnu/gcc/gcc-13.3.0/gcc-13.3.0.tar.xz"
GCC_SHA = "0845e9621c9543a13f484e94584a49ffc0129970e9914624235fc1d061a0c083"     # Homebrew gcc@13 13.3.0 formula
GCC_PATCH_URL = ("https://raw.githubusercontent.com/Homebrew/formula-patches/bda0faddfbfb392e7b9c9101056b2c5ab2500508/"
                 "gcc/gcc-13.3.0.diff")
GCC_PATCH_SHA = "c5e9236430ef6edbdda7de9ac70bf79e21628077a48322cec7f3f064ccfc243d"
QUADMATH_MARK = b"/gcc-13.3.0/build/aarch64-apple-darwin23/libquadmath/"
KEEP_BIN = ("python", "python3", "python3.10")            # every other runtime/bin entry is a script with a build path
STRIP_DIRS = ("include", os.path.join("lib", "pkgconfig"), "share",
              os.path.join("lib", "python3.10", "config-3.10-darwin"),
              os.path.join("lib", "python3.10", "site-packages", "torch", "include"))
STRIP_SUFFIX = (".a",)
MACHO_MAGIC = (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xfe\xed\xfa\xcf")
ICON_SIZES = [(16, ""), (32, "@2x", 16), (32, ""), (64, "@2x", 32), (128, ""), (256, "@2x", 128), (256, ""),
              (512, "@2x", 256), (512, ""), (1024, "@2x", 512)]


def say(*a):
    print("[build-mac]", *a, file=sys.stderr, flush=True)   # stdout carries only values a caller captures


def run(cmd, **kw):
    say(" ".join(cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}")
    return r


def is_macho(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) in MACHO_MAGIC
    except OSError:
        return False


def tree_size(p):
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs
               if not os.path.islink(os.path.join(a, f)))


def fetch_checked(url, sha, name, what):
    """-> the cached file `name`, downloaded once and kept only when its sha256 is `sha`."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name)
    if not (os.path.isfile(p) and W.sha(p) == sha):
        say("downloading", url)
        urllib.request.urlretrieve(url, p + ".part")
        if W.sha(p + ".part") != sha:
            os.remove(p + ".part")
            raise SystemExit(f"{what}: sha256 mismatch")
        os.replace(p + ".part", p)
    return p


def fetch_python():
    return fetch_checked(PBS_URL, PBS_SHA, os.path.basename(PBS_URL).replace("%2B", "+"),
                         "python-build-standalone (aarch64)")


def mac_pins():
    """requirements.in + constraints.txt with the CUDA local version taken off (torch / torchvision 2.10.0+cu128 ->
    2.10.0): the same versions, from PyPI. Nothing else may differ."""
    out = {}
    for name in ("requirements.in", "constraints.txt"):
        src = open(os.path.join(PKG, name), encoding="utf-8").read()
        dst = src.replace("+cu128", "")
        changed = [l for l in src.splitlines() if "+cu128" in l]
        if not any(l.startswith("torch==") for l in changed):
            raise SystemExit(f"{name}: no torch==...+cu128 line -- check the pins")
        p = os.path.join(OUTDIR, "pins_" + name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(dst)
        say(f"{name}: CUDA suffix taken off {len(changed)} line(s): {', '.join(changed)}")
        out[name] = p
    return out["requirements.in"], out["constraints.txt"]


def main():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("build_package_mac.py runs on an Apple silicon Mac (the GitHub Actions macOS runner)")
    t0 = time.time()
    sys.path.insert(0, REPO)
    from vathography_app import __version__
    man = json.load(open(os.path.join(REPO, "vathography_app", "model_manifest.json"), encoding="utf-8"))
    tp = man["third_party_macos"]
    os.makedirs(WHEELS, exist_ok=True)
    if os.path.isdir(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.makedirs(RES)
    os.makedirs(os.path.join(CONTENTS, "MacOS"))
    tgz = fetch_python()
    run(["tar", "-xzf", tgz, "-C", OUTDIR])
    runtime = os.path.join(RES, "runtime")
    os.replace(os.path.join(OUTDIR, "python"), runtime)
    py = os.path.join(runtime, "bin", "python3.10")
    site = os.path.join(runtime, "lib", "python3.10", "site-packages")
    req, con = mac_pins()
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    pip = [py, "-s", "-m", "pip"]
    # 1 wheelhouse (binary wheels only, resolved by the TARGET runtime: cp310 macosx arm64) + 2 install offline
    run(pip + ["download", "-d", WHEELS, "--only-binary=:all:", "-r", req, "-c", con], env=env)
    for e in tp:                               # the wheels the first start downloads = the wheels the build resolved
        w = os.path.join(WHEELS, e["file"])
        if not os.path.isfile(w) or os.path.getsize(w) != e["size"] or W.sha(w) != e["sha256"]:
            raise SystemExit(f"{e['file']}: pip did not resolve the manifest's third_party_macos pin (file / size / "
                             f"sha256) -- the first start would install a different {e['name']} than the build")
    run(pip + ["install", "--no-index", "--find-links", WHEELS, "--no-warn-script-location", "-r", req, "-c", con],
        env=env)
    # 2b the LGPL PyAV replaces PyPI's (which bundles libx264 + libx265)
    lgpl = glob.glob(os.path.join(AV_LGPL_DIR, "av-17.1.0-*.whl"))
    if not lgpl:
        raise SystemExit("no LGPL PyAV wheel: run packaging/build_av_lgpl_mac.py first")
    run(pip + ["install", "--no-index", "--no-deps", "--force-reinstall", "--no-warn-script-location", lgpl[0]],
        env=env)
    av_libs = os.path.join(site, "av", ".dylibs")
    libs = os.listdir(av_libs) if os.path.isdir(av_libs) else []
    gpl = [f for f in libs if f.lower().startswith(W.GPL_LIBS)]
    avcodec = [f for f in libs if f.startswith("libavcodec")]
    blob = open(os.path.join(av_libs, avcodec[0]), "rb").read() if avcodec else b""
    if (gpl or not avcodec or W.AV_FFMPEG_LICENCE not in blob or b"GPL version 2 or later" in blob
            or W.AV_FFMPEG_VERSION.encode() not in blob):
        raise SystemExit(f"PyAV's FFmpeg is not the LGPL build (libs: {libs}; GPL libraries: {gpl})")
    dist = glob.glob(os.path.join(site, "av-17.1.0.dist-info"))[0]
    if os.path.isfile(os.path.join(dist, "direct_url.json")):        # a file:// URL of the build machine
        os.remove(os.path.join(dist, "direct_url.json"))
        rec = os.path.join(dist, "RECORD")
        keep = [l for l in open(rec, encoding="utf-8").read().splitlines()
                if not l.split(",")[0].endswith("/direct_url.json")]
        with open(rec + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(keep) + "\n")
        os.replace(rec + ".tmp", rec)
    say(f"PyAV: {W.AV_FFMPEG_VERSION}, {W.AV_FFMPEG_LICENCE.decode()}, no GPL library ({len(libs)} dylibs)")
    freeze = subprocess.run(pip + ["freeze"], env=env, capture_output=True, text=True).stdout.split()
    tcl = subprocess.run([py, "-E", "-s", "-c", "import tkinter; print(tkinter.Tcl().eval('info tclversion'))"],
                         capture_output=True, text=True).stdout.strip()
    if not re.fullmatch(r"\d+\.\d+", tcl):
        raise SystemExit(f"the runtime's Tcl version could not be read: {tcl!r}")
    say(f"runtime Tcl/Tk {tcl}")
    # 3 strip what never runs
    removed = 0
    for d in STRIP_DIRS:
        p = os.path.join(runtime, d)
        if os.path.isdir(p):
            removed += tree_size(p)
            shutil.rmtree(p)
    for a, ds, fs in os.walk(runtime):
        for f in fs:
            if f.endswith(STRIP_SUFFIX):
                removed += os.path.getsize(os.path.join(a, f))
                os.remove(os.path.join(a, f))
        for d in [d for d in ds if d == "__pycache__"]:
            shutil.rmtree(os.path.join(a, d))
            ds.remove(d)
    bindir = os.path.join(runtime, "bin")
    n_bin = 0
    for f in os.listdir(bindir):
        if f not in KEEP_BIN:
            p = os.path.join(bindir, f)
            removed += 0 if os.path.islink(p) else os.path.getsize(p)
            os.remove(p)
            n_bin += 1
    n_tp = 0                                   # Part T / Q4: PyTorch + OpenCV out of the zip
    for d in os.listdir(site):
        p = os.path.join(site, d)
        if d in W.THIRD_PARTY_TOPS or (d.startswith(W.THIRD_PARTY_DIST) and d.endswith(".dist-info")):
            removed += tree_size(p)
            shutil.rmtree(p)
            n_tp += 1
    if n_tp != len(W.THIRD_PARTY_TOPS) + len(W.THIRD_PARTY_DIST):
        raise SystemExit(f"expected torch, functorch, torchgen, cv2 + 2 dist-infos in site-packages, removed {n_tp}")
    n_pip = 0
    for d in os.listdir(site):
        p = os.path.join(site, d)
        if d in W.PIP_DIRS or (d.startswith(W.PIP_DIST) and d.endswith(".dist-info")):
            shutil.rmtree(p)
            n_pip += 1
        elif d == "distutils-precedence.pth" or (d.endswith(".whl") and os.path.getsize(p) == 0):
            os.remove(p)
            n_pip += 1
    for p in glob.glob(os.path.join(runtime, "lib", "python3.10", "ensurepip", "_bundled", "*.whl")):
        removed += os.path.getsize(p)
        os.remove(p)
        n_pip += 1
    n_unused = 0
    for d in os.listdir(site):
        if d in W.UNUSED_DIRS or (d.startswith(W.UNUSED_DIST) and d.endswith(".dist-info")):
            p = os.path.join(site, d)
            removed += tree_size(p)
            shutil.rmtree(p)
            n_unused += 1
    if n_unused != len(W.UNUSED_DIRS) + len(W.UNUSED_DIST):
        raise SystemExit(f"expected {W.UNUSED_DIRS} + their dist-info in site-packages, removed {n_unused}")
    # tkinterdnd2 loads <platform>-tcl9 under Tcl 9 when it exists, else <platform> (TkinterDnD.py): keep that one
    tkdnd = os.path.join(site, "tkinterdnd2", "tkdnd")
    keep = "osx-arm64-tcl9" if int(tcl.split(".")[0]) >= 9 else "osx-arm64"
    if keep not in os.listdir(tkdnd):
        raise SystemExit(f"tkdnd/{keep} not found -- check the tkinterdnd2 version")
    for d in [d for d in os.listdir(tkdnd) if d != keep]:
        removed += tree_size(os.path.join(tkdnd, d))
        shutil.rmtree(os.path.join(tkdnd, d))
    # diffusers: the non-commercial files out + the one import line (identical to build_package.py, Windows)
    for rel in W.NONCOMMERCIAL_PATHS:
        p = os.path.join(site, rel)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)
        else:
            raise SystemExit(f"{rel} not found -- check the diffusers version")
    ti = os.path.join(site, W.TRANSFORMERS_INIT)
    src = open(ti, "rb").read()
    eol = b"\r\n" if b"\r\n" in src else b"\n"
    lines = src.split(eol)
    if lines.count(W.BRIA_IMPORT_LINE) != 1:
        raise SystemExit(f"{W.TRANSFORMERS_INIT}: the transformer_bria_fibo import line is not there exactly once")
    lines[lines.index(W.BRIA_IMPORT_LINE)] = W.BRIA_CHANGED_NOTE
    with open(ti + ".tmp", "wb") as f:
        f.write(eol.join(lines))
    os.replace(ti + ".tmp", ti)
    left_bria = [os.path.relpath(os.path.join(a, f), site) for a, _, fs in os.walk(os.path.join(site, "diffusers"))
                 for f in fs if f.endswith(".py") and re.search(rb"^\s*from \.+[\w.]*transformer_bria_fibo import",
                                                                open(os.path.join(a, f), "rb").read(), re.M)]
    if left_bria:
        raise SystemExit("an import of the removed transformer_bria_fibo is left:\n  " + "\n  ".join(left_bria))
    nc = [os.path.relpath(os.path.join(a, f), runtime) for a, _, fs in os.walk(runtime) for f in fs if f.endswith(".py")
          and W.NONCOMMERCIAL_RX.search(open(os.path.join(a, f), "rb").read(3000).decode("utf-8", "replace"))]
    if nc:
        raise SystemExit("non-commercial licence header left in the runtime:\n  " + "\n  ".join(nc[:20]))
    # STOP on any leftover: by name, then by content over every Mach-O of the runtime
    left = []
    for a, ds, fs in os.walk(runtime):
        for n in ds + fs:
            low = n.lower()
            top_pkg = os.path.samefile(a, site) and (low in W.THIRD_PARTY_TOPS or low in W.PIP_DIRS
                                                    or low in W.UNUSED_DIRS)
            if low.startswith(W.LEFTOVER_PREFIX) or top_pkg or low.endswith((".pth", ".whl")) \
                    or low.startswith(W.GPL_LIBS) \
                    or (low.startswith(W.PIP_DIST + W.UNUSED_DIST) and low.endswith(".dist-info")):
                left.append(os.path.join(a, n))
    if left:
        raise SystemExit("left in the runtime, must not ship:\n  " + "\n  ".join(left[:20]))
    hits, n_macho = [], 0
    for a, _, fs in os.walk(runtime):
        for n in fs:
            p = os.path.join(a, n)
            if os.path.islink(p) or not is_macho(p):
                continue
            n_macho += 1
            blob = open(p, "rb").read()
            found = [m.decode() for m in W.PROPRIETARY_MARKERS if m in blob]
            if found:
                hits.append(f"{os.path.relpath(p, runtime)}: {', '.join(found)}")
    if hits:
        raise SystemExit("proprietary marker in a binary of the package:\n  " + "\n  ".join(hits[:20]))
    say(f"stripped {removed / 1e9:.2f} GB ({n_bin} bin scripts, PyTorch + OpenCV ({n_tp} folders, downloaded at "
        f"first start), pip + setuptools ({n_pip} items), tkdnd kept: {keep}); no leftover by name, no proprietary "
        f"marker in {n_macho} Mach-O files")
    # 4 the app, the launcher, the icon, Info.plist, the texts
    appdir = os.path.join(RES, "app", "vathography_app")
    shutil.copytree(os.path.join(REPO, "vathography_app"), appdir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp"))
    run(["clang", "-O2", "-arch", "arm64", "-mmacosx-version-min=14.0", "-Wall",
         os.path.join(MAC, "launcher.c"), "-o", os.path.join(CONTENTS, "MacOS", NAME)])
    iconset = os.path.join(OUTDIR, "vathography_app.iconset")
    os.makedirs(iconset)
    draw = ("import sys; sys.path.insert(0, sys.argv[1]); import make_icon\n"
            "for px, name in zip(map(int, sys.argv[3::2]), sys.argv[4::2]):\n"
            "    make_icon.draw(px).save(sys.argv[2] + '/' + name)")
    names = []
    for s in ICON_SIZES:
        px, tag = s[0], s[1]
        base = s[2] if len(s) > 2 else px
        names += [str(px), f"icon_{base}x{base}{tag}.png"]
    run([py, "-E", "-s", "-c", draw, PKG, iconset] + names)
    run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(RES, "vathography_app.icns")])
    shutil.rmtree(iconset)
    info = open(os.path.join(MAC, "Info.plist"), encoding="utf-8").read().replace("{version}", __version__)
    plistlib.loads(info.encode("utf-8"))                          # a malformed plist stops the build here
    with open(os.path.join(CONTENTS, "Info.plist"), "w", encoding="utf-8") as f:
        f.write(info)
    release_texts = all(os.path.isfile(os.path.join(MAC, f)) for f in ("README.txt", "NOTICE.txt"))
    if release_texts:
        readme, notice = os.path.join(MAC, "README.txt"), os.path.join(MAC, "NOTICE.txt")
    else:                                      # TEST build: the Windows texts are wrong on a Mac
        readme = os.path.join(OUTDIR, "README.txt")
        with open(readme, "w", encoding="utf-8") as f:
            f.write("TEST BUILD of Vathography App for macOS - not for distribution. The macOS README and notices are "
                    "not written yet; NOTICE.txt is the Windows text.\n")
        notice = os.path.join(OUTDIR, "NOTICE.txt")
        with open(notice, "w", encoding="utf-8") as f:
            f.write("DRAFT - the Windows NOTICE.txt, not yet checked for the macOS package.\n\n"
                    + open(os.path.join(PKG, "NOTICE.txt"), encoding="utf-8").read())
    for d in (FOLDER, RES):
        shutil.copy(W.APP_LICENSE, os.path.join(d, "LICENSE"))
        shutil.copy(notice, os.path.join(d, "NOTICE.txt"))
    shutil.copy(readme, os.path.join(FOLDER, "README.txt"))
    shutil.copytree(os.path.join(PKG, "licences_macos"), os.path.join(FOLDER, "licences"))   # the Mac set (W2)
    src_dir = os.path.join(FOLDER, "source")
    os.makedirs(src_dir)
    if W.sha(FFMPEG_SRC) != W.FFMPEG_SRC_SHA:
        raise SystemExit(f"{FFMPEG_SRC}: not the FFmpeg n8.1.1 archive the dylibs were built from")
    shutil.copy(FFMPEG_SRC, os.path.join(src_dir, "ffmpeg-n8.1.1.tar.gz"))
    import tarfile
    with tarfile.open(FFMPEG_SRC) as t:
        m = next(x for x in t.getmembers() if x.name.endswith("/COPYING.LGPLv2.1"))
        with open(os.path.join(src_dir, "COPYING.LGPLv2.1"), "wb") as f:
            f.write(t.extractfile(m).read())
    quad = glob.glob(os.path.join(runtime, "lib", "python3.10", "site-packages", "scipy", ".dylibs", "libquadmath*"))
    if len(quad) != 1 or QUADMATH_MARK not in open(quad[0], "rb").read():
        raise SystemExit(f"SciPy's libquadmath is not the GCC 13.3.0 build whose source source/ carries: {quad}")
    shutil.copy(fetch_checked(GCC_URL, GCC_SHA, "gcc-13.3.0.tar.xz", "GCC 13.3.0"),
                os.path.join(src_dir, "gcc-13.3.0.tar.xz"))
    shutil.copy(fetch_checked(GCC_PATCH_URL, GCC_PATCH_SHA, "gcc-13.3.0.diff", "Homebrew's GCC 13.3.0 patch"),
                os.path.join(src_dir, "gcc-13.3.0.diff"))
    clang = subprocess.run(["clang", "--version"], capture_output=True, text=True).stdout.splitlines()
    if not clang or "clang" not in clang[0]:
        raise SystemExit(f"clang --version gave no version line: {clang}")
    text = open(os.path.join(MAC, "SOURCE_README.txt"), encoding="utf-8").read()
    if text.count("{CLANG_VERSION}") != 1:
        raise SystemExit("packaging/macos/SOURCE_README.txt: expected one {CLANG_VERSION}")
    with open(os.path.join(src_dir, "README.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace("{CLANG_VERSION}", clang[0].strip()))
    # precompiled once, here: with every .pyc present and valid, Python never writes into the signed bundle
    r = subprocess.run([py, "-E", "-s", "-m", "compileall", "-q", "-j", "0", os.path.join(runtime, "lib"),
                        os.path.join(RES, "app")], capture_output=True, text=True)
    say(f"compileall rc {r.returncode}" + (f" ({len(r.stdout.splitlines())} lines of output; files Python cannot "
                                           "compile are never imported)" if r.returncode else ""))
    used = sorted(w for w in os.listdir(WHEELS) if w.endswith(".whl"))
    installed = {l.split("==")[0].lower().replace("_", "-") for l in freeze if "==" in l}
    wheels = [dict(file=w, sha256=W.sha(os.path.join(WHEELS, w))) for w in used
              if w.split("-")[0].lower().replace("_", "-") in installed and w.split("-")[0].lower() != "av"]
    wheels.append(dict(file=os.path.basename(lgpl[0]), sha256=W.sha(lgpl[0]),
                       built_from="PyAV 17.1.0 sdist + FFmpeg 8.1.1 (tag n8.1.1), LGPL-2.1-or-later, minimal "
                                  "configure (packaging/build_av_lgpl_mac.py)"))
    build_info = dict(app=NAME, version=__version__, platform="macOS arm64 (Apple silicon), macOS 14 or newer",
                      built=time.strftime("%Y-%m-%d %H:%M"), test_build=not release_texts,
                      python=dict(source=PBS_URL, sha256=PBS_SHA, tcl_tk=tcl),
                      note="freeze and wheels record the set the build resolved. Not in this package: PyTorch and "
                           "OpenCV (third_party_macos: downloaded from PyPI at first start), pip and setuptools, "
                           "hf-xet",
                      freeze=freeze, wheels=wheels,
                      third_party=[{k: e[k] for k in ("name", "version", "url", "size", "sha256", "folder")}
                                   for e in tp])
    with open(os.path.join(FOLDER, "BUILD_INFO.json"), "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=1)
    # 5 sign ad hoc (Q2): every Mach-O inside the bundle, then the bundle; verify
    run(["xattr", "-cr", FOLDER])
    machos = [os.path.join(a, f) for a, _, fs in os.walk(CONTENTS) for f in fs
              if not os.path.islink(os.path.join(a, f)) and is_macho(os.path.join(a, f))
              and os.path.join(a, f) != os.path.join(CONTENTS, "MacOS", NAME)]
    say(f"signing {len(machos)} Mach-O files ad hoc")
    for p in sorted(machos, key=lambda p: -p.count(os.sep)):
        r = subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", p], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"codesign failed on {os.path.relpath(p, FOLDER)}: {r.stderr.strip()}")
    run(["codesign", "--force", "--sign", "-", "--timestamp=none", APP])
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", APP])
    size = tree_size(FOLDER)
    say(f"package folder: {size / 1e9:.2f} GB, {len(freeze)} packages")
    # 6 the zip: ditto, no resource forks / extended attributes (no __MACOSX, no ._ files); unpacked again + verified
    z = os.path.join(BUILD, f"{NAME.replace(' ', '_')}_{__version__}_macos{'' if release_texts else '_TEST'}.zip")
    if os.path.exists(z):
        os.remove(z)
    run(["ditto", "-c", "-k", "--norsrc", "--noextattr", "--keepParent", FOLDER, z + ".part"])
    os.replace(z + ".part", z)
    bad = [n for n in zipfile.ZipFile(z).namelist()
           if n.startswith("__MACOSX") or "/._" in n or n.endswith(".DS_Store") or "__pycache__/" in n
           and not n.startswith(f"{NAME}/{NAME}.app/")]
    if bad:
        raise SystemExit("the zip carries Finder / AppleDouble files:\n  " + "\n  ".join(bad[:20]))
    chk = os.path.join(BUILD, "mac_unzip_check")
    shutil.rmtree(chk, ignore_errors=True)
    os.makedirs(chk)
    run(["ditto", "-x", "-k", z, chk])
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", os.path.join(chk, NAME, NAME + ".app")])
    shutil.rmtree(chk)
    say(f"zip: {z} {os.path.getsize(z) / 1e9:.2f} GB sha256 {W.sha(z)}")
    say(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
