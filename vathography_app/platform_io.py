# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""platform_io.py — everything that differs between Windows and macOS, standard library only (the window imports it;
it never imports torch). The Windows branches serve Windows 10 / 11; the macOS branches
are for Apple Silicon, macOS 14 or newer.

  IS_WIN / IS_MAC          the platform
  data_home(install_home)  where models, third_party, settings and the run log live: beside the app on Windows; in
                           ~/Library/Application Support/Vathography App when the app runs from a macOS .app bundle
                           (a downloaded bundle runs from a read-only copy until it is moved)
  no_window()              Popen / run keyword arguments that hide a console window (Windows only)
  worker_python()          the interpreter for the worker process
  ram_gb()                 total physical memory, GB
  pid_alive(pid)           the process still runs (the install lock)
  safe_rename(tmp, dst)    tmp -> dst, NEVER replacing an existing dst (FileExistsError)
  same_volume(a, b)        two paths on the same disk (the disk-space checks)
  fonts()                  (serif, sans, mono) family names
  os_line()                one line naming the OS for the troubleshooting log
  mac_problem()            None, or the sentence why this Mac cannot run the app
"""
import errno
import os
import platform
import shutil
import subprocess
import sys

IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
MAC_MIN = (14, 0)                     # bf16 on Metal (PyTorch MPS) needs macOS 14
APP_SUPPORT_NAME = "Vathography App"


def in_mac_bundle(path):
    """True when `path` lies inside a macOS application bundle (<name>.app/Contents/...)."""
    parts = os.path.abspath(path).split(os.sep)
    return any(p.lower().endswith(".app") and i + 1 < len(parts) and parts[i + 1] == "Contents"
               for i, p in enumerate(parts))


def data_home(install_home):
    """The writable home of the app's data (models, third_party, settings, run log)."""
    if IS_MAC and in_mac_bundle(install_home):
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", APP_SUPPORT_NAME)
    return install_home


def no_window():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WIN else {}


def worker_python():
    exe = sys.executable
    if IS_WIN and os.path.basename(exe).lower() == "pythonw.exe":
        exe = os.path.join(os.path.dirname(exe), "python.exe")
    return exe


def ram_gb():
    if IS_WIN:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1e9
    if IS_MAC:
        r = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip()) / 1e9
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9


def chip_name():
    """macOS: the chip ('Apple M4 Pro'); elsewhere ''."""
    if not IS_MAC:
        return ""
    try:
        r = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def pid_alive(pid):
    if pid == os.getpid():
        return True
    if IS_WIN:
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
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:                           # it runs, under another user
        return True
    except OSError:
        return True


def safe_rename(tmp, dst):
    """tmp -> dst; an existing dst is never replaced: FileExistsError, tmp left in place.
    Windows: os.rename (Windows refuses an existing target). POSIX: os.rename REPLACES silently, so a hard link
    (fails if dst exists) then tmp removed; a volume without hard links (exFAT, some network shares): dst created
    exclusively (O_EXCL), tmp's bytes copied in, tmp removed."""
    if IS_WIN:
        os.rename(tmp, dst)
        return
    try:
        os.link(tmp, dst)
    except FileExistsError:
        raise
    except OSError as e:
        if e.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK, errno.ENOSYS, errno.EACCES):
            raise
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)   # FileExistsError when taken
        try:
            with os.fdopen(fd, "wb") as out, open(tmp, "rb") as src:
                shutil.copyfileobj(src, out, 1 << 22)
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            try:
                os.remove(dst)
            except OSError:
                pass
            raise
    os.remove(tmp)


def _existing(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        q = os.path.dirname(p)
        if q == p:
            break
        p = q
    return p


def same_volume(a, b):
    if IS_WIN:
        return (os.path.splitdrive(os.path.abspath(a))[0].lower()
                == os.path.splitdrive(os.path.abspath(b))[0].lower())
    try:
        return os.stat(_existing(a)).st_dev == os.stat(_existing(b)).st_dev
    except OSError:
        return True


def fonts():
    """(serif, sans, mono): the site's fonts as each OS renders them (Palatino + the system sans)."""
    if IS_MAC:
        return "Palatino", "Helvetica Neue", "Menlo"
    return "Palatino Linotype", "Segoe UI", "Consolas"


def os_line():
    if IS_WIN:
        return f"Windows: {platform.platform()}"
    if IS_MAC:
        return f"macOS: {platform.mac_ver()[0]} ({platform.machine()})"
    return f"OS: {platform.platform()}"


def mac_version():
    try:
        return tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
    except ValueError:
        return (0, 0)


def mac_problem():
    """None when this Mac can run the app, else one sentence."""
    if platform.machine() != "arm64":
        return "The app needs a Mac with Apple silicon (M1 or later); this Mac has an Intel processor."
    v = mac_version()
    if v < MAC_MIN:
        return (f"The app needs macOS {MAC_MIN[0]} or newer; this Mac runs macOS {platform.mac_ver()[0]}. "
                "Update macOS in System Settings, then try again.")
    return None
