# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""third_party.py — PyTorch + OpenCV, downloaded at first start from their publishers, installed beside the app.
Standard library only: it runs before either package exists.

  <APP_HOME>\third_party\<folder>\     one folder per manifest `third_party` entry, found by Python through ONE .pth file
                                       the package build writes into site-packages (two relative lines)
  <folder>\.installed                  name + version + sha256, written LAST: an install is valid only when this file
                                       matches the pin, never because the folder exists

install() verifies the wheel against its pin, unpacks it into <folder>.tmp on the same drive (leaving out C headers,
import libraries, debug symbols, the two NVIDIA DLLs NVIDIA does not list as redistributable and OpenCV's video FFmpeg),
writes .installed, then renames the folder into place. A crash at any point leaves no valid folder under the final name.
One install at a time: a lock file names the process holding it; a second window refuses while that process lives.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import time
import zipfile

from . import models

HOME = os.path.join(models.APP_HOME, "third_party")
MARK = ".installed"
LOCK = os.path.join(HOME, ".install.lock")
BLOCK = 1 << 22                                   # progress + Stop every 4 MiB, also inside one large DLL
SKIP_DIRS = ("torch/include/",)
SKIP_SUFFIX = (".lib", ".pdb")
SKIP_FILES = ("torch/lib/nvperf_host.dll", "torch/lib/cusolverMg64_11.dll")   # not in NVIDIA's redistribution lists
SKIP_PREFIX = ("cv2/opencv_videoio_ffmpeg",)      # OpenCV's own FFmpeg: video I/O only, never used by the app


class Cancelled(Exception):
    pass


class Busy(Exception):
    pass


def entries():
    with open(models.MANIFEST, encoding="utf-8") as f:
        return json.load(f).get("third_party", [])


def folder(entry, home=None):
    return os.path.join(home or HOME, entry["folder"])


def skipped(name):
    """True for a wheel member left out at install (never needed to run)."""
    low = name.lower()
    return (name.startswith(SKIP_DIRS) or low.endswith(SKIP_SUFFIX) or name in SKIP_FILES
            or name.startswith(SKIP_PREFIX))


def _mark(path):
    try:
        with open(os.path.join(path, MARK), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def installed(entry, home=None):
    """The pinned version is installed: its folder carries a .installed that matches name, version AND sha256."""
    m = _mark(folder(entry, home))
    return bool(m) and all(m.get(k) == entry[k] for k in ("name", "version", "sha256"))


def _found_elsewhere(entry, home=None):
    """Development environment: the module is importable from outside third_party\\ (find_spec imports nothing)."""
    try:
        spec = importlib.util.find_spec(entry["module"])
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.origin:
        return False
    root = os.path.normcase(os.path.abspath(home or HOME)) + os.sep
    return not os.path.normcase(os.path.abspath(spec.origin)).startswith(root)


def over_old_version():
    """The package was unzipped over a version that still carried PyTorch / OpenCV in its runtime (0.1.6 and earlier):
    those leftovers shadow third_party\\. -> True only in the package layout (<APP_HOME>\\runtime exists)."""
    site = os.path.join(models.APP_HOME, "runtime", "Lib", "site-packages")
    return os.path.isdir(site) and any(os.path.isdir(os.path.join(site, d)) for d in ("torch", "cv2"))


def missing(home=None):
    """The entries still to install (cheap: reads .installed files, imports nothing)."""
    return [e for e in entries() if not (installed(e, home) or _found_elsewhere(e, home))]


def ready(home=None):
    return not missing(home)


def not_ready_sentence(miss):
    names = " and ".join(e["name"] for e in miss)
    return (f"{names} {'is' if len(miss) == 1 else 'are'} not installed yet. Open the app's window, click Download "
            f"at the top, then try again.")


def load_check():
    """Really imports torch and cv2 before any work (pipeline.py swallows a cv2 failure). -> None or a sentence.
    OpenCV reads OPENCV_IO_ENABLE_OPENEXR once, at its first import: this import is now the first one, so it sets the
    switch relief_app.py relies on (it reads EXR exports with cv2.imread)."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    for mod, name in (("torch", "PyTorch"), ("cv2", "OpenCV")):
        try:
            __import__(mod)
        except Exception as ex:                   # noqa: BLE001 -- any load failure ends in one sentence
            hint = ""
            if isinstance(ex, OSError) or "DLL" in str(ex):
                hint = (" Windows could not load one of its libraries: installing the Microsoft Visual C++ "
                        "Redistributable (x64) usually fixes this.")
            return f"{name} is installed but does not load ({type(ex).__name__}: {ex}).{hint}"
    return None


def _pid_alive(pid):
    if pid == os.getpid():
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259     # STILL_ACTIVE
    except (AttributeError, OSError):
        return True                               # cannot tell: refuse rather than install twice


def _lock(home):
    lock = os.path.join(home, os.path.basename(LOCK))
    for _ in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                pid = int(open(lock, encoding="ascii").read().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                raise Busy("Another Vathography App window is installing PyTorch or OpenCV; wait until it finishes.")
            os.remove(lock)                       # left by a process that no longer runs
    raise Busy("The install lock could not be taken.")


def _retry(fn, *a):
    """Defender and the search indexer briefly hold new DLLs: retry a rename / delete before giving up."""
    for i in range(20):
        try:
            return fn(*a)
        except PermissionError:
            if i == 19:
                raise
            time.sleep(0.25 + 0.05 * i)


def _rmtree(path):
    if os.path.isdir(path):
        _retry(shutil.rmtree, path)


def _sha(path, cancel=None, progress=None):
    h, done, size = hashlib.sha256(), 0, os.path.getsize(path)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            h.update(b)
            done += len(b)
            if progress:
                progress("check", done, size)
    return h.hexdigest()


def clean(home=None):
    """Our leftovers: any <folder>.tmp, and any entry folder without a valid .installed (os.replace refuses an
    existing folder, even an empty one)."""
    home = home or HOME
    if not os.path.isdir(home):
        return
    names = {e["folder"] for e in entries()}
    for d in os.listdir(home):
        p = os.path.join(home, d)
        if not os.path.isdir(p):
            continue
        if d.endswith(".tmp") and d[:-4] in names:
            _rmtree(p)
        elif d in names and not installed(next(e for e in entries() if e["folder"] == d), home):
            _rmtree(p)


def unpacked_bytes(wheel):
    """Bytes the install writes (after the filter)."""
    with zipfile.ZipFile(wheel) as z:
        return sum(i.file_size for i in z.infolist() if not i.is_dir() and not skipped(i.filename))


def install(entry, wheel, progress=None, cancel=None, home=None, delete_wheel=False):
    """Install one pinned wheel. progress(phase, done, total) with phase 'check' | 'unpack'; cancel = threading.Event.
    delete_wheel: True only when download.fetch put the wheel there (a wheel named by the caller is never touched).
    -> the installed folder."""
    home = home or HOME
    os.makedirs(home, exist_ok=True)
    lock = _lock(home)
    final, tmp = folder(entry, home), folder(entry, home) + ".tmp"
    try:
        if installed(entry, home):                # already valid (a second call, another window finished first)
            return final
        if os.path.getsize(wheel) != entry["size"] or _sha(wheel, cancel, progress) != entry["sha256"]:
            raise RuntimeError(f"{os.path.basename(wheel)} does not match the pinned {entry['name']} "
                               f"{entry['version']}; nothing was installed.")
        clean(home)
        os.makedirs(tmp)
        root = os.path.abspath(tmp) + os.sep
        with zipfile.ZipFile(wheel) as z:
            todo = [i for i in z.infolist() if not i.is_dir() and not skipped(i.filename)]
            total, done = sum(i.file_size for i in todo), 0
            for i in todo:
                dst = os.path.abspath(os.path.join(tmp, *i.filename.split("/")))
                if not dst.startswith(root):
                    raise RuntimeError(f"{os.path.basename(wheel)}: unsafe path {i.filename!r}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(i) as src, open(dst, "wb") as out:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        b = src.read(BLOCK)
                        if not b:
                            break
                        out.write(b)
                        done += len(b)
                        if progress:
                            progress("unpack", done, total)
        with open(os.path.join(tmp, MARK), "w", encoding="utf-8") as f:   # LAST: the folder becomes valid
            json.dump({k: entry[k] for k in ("name", "version", "sha256")}, f)
        _retry(os.replace, tmp, final)
        for d in os.listdir(home):                # older pins of the same package, only folders we installed
            p = os.path.join(home, d)
            m = _mark(p) if os.path.isdir(p) and p != final else None
            if m and m.get("name") == entry["name"]:
                _rmtree(p)
        if delete_wheel:
            try:
                os.remove(wheel)
            except OSError:
                pass
        return final
    except BaseException:
        try:
            _rmtree(tmp)
        except OSError:
            pass                                  # clean() removes it at the next install
        raise
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass
