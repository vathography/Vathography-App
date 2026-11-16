r"""smoke_mac.py — checks on the BUILT macOS zip, on the GitHub Actions runner (no GPU there: no depth map is made).
The macOS counterpart of gate G2's Part-T checks (gates/g2_package.py) for the parts a runner can test:
  S1 the zip unpacks (ditto) and the bundle's signature verifies
  S2 the window's code imports with neither torch nor cv2; neither is importable from a fresh unpack
  S3 `depth` is refused with one sentence, no traceback
  S4 PyTorch + OpenCV installed OFFLINE from the wheels the build resolved, through the hidden `install-third-party`
     verb (the window's install()), into ~/Library/Application Support/Vathography App/third_party
  S5 after S4: torch + cv2 load from that folder; torch has MPS built in; every module of vathography_app imports (CPU)
  S6 the bundle's own executable starts the window and is still running after 20 s (then closed)
  S7 nothing was written into the bundle by S2-S6 (its signature still verifies)
Usage (macOS): python3 packaging/smoke_mac.py <zip> <wheels folder>
"""
import os
import shutil
import signal
import subprocess
import sys
import time

zpath, wheels = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
NAME = "Vathography App"
work = os.path.join(os.path.dirname(zpath), "smoke_unzip")
support = os.path.join(os.path.expanduser("~"), "Library", "Application Support", NAME)
res = []


def check(name, ok, info=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, str(info)[-800:], flush=True)


shutil.rmtree(work, ignore_errors=True)
shutil.rmtree(support, ignore_errors=True)               # a runner is fresh; this makes a re-run fresh too
os.makedirs(work)
r = subprocess.run(["ditto", "-x", "-k", zpath, work], capture_output=True, text=True)
app = os.path.join(work, NAME, NAME + ".app")
contents = os.path.join(app, "Contents")
py = os.path.join(contents, "Resources", "runtime", "bin", "python3.10")
appdir = os.path.join(contents, "Resources", "app")


def verify():
    v = subprocess.run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", app], capture_output=True, text=True)
    return v.returncode == 0, v.stderr.strip()


ok, info = verify()
check("S1 unzipped, signature verifies", r.returncode == 0 and ok, info)
if not res[-1]:
    sys.exit(1)
if os.environ.get("SEED_DEFECT") == "smoke":             # must-fire test run: smoke has to go red
    seeded = os.path.join(appdir, "vathography_app", "outnames.py")
    for f in [seeded] + [os.path.join(appdir, "vathography_app", "__pycache__", n)
                         for n in os.listdir(os.path.join(appdir, "vathography_app", "__pycache__"))
                         if n.startswith("outnames.")]:
        os.remove(f)
    print("SEEDED DEFECT smoke: removed vathography_app/outnames.py (+ its .pyc) from the unzipped bundle; "
          "S5 'every module imports' and S7 'signature' must FAIL", flush=True)


def prun(args, timeout=900):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    return subprocess.run([py, "-E", "-s"] + args, cwd=appdir, capture_output=True, text=True, env=env, timeout=timeout)


r = prun(["-c", "import vathography_app.gui, sys; print('torch' in sys.modules, 'cv2' in sys.modules)"])
check("S2 the window imports with neither torch nor cv2", r.returncode == 0 and r.stdout.strip() == "False False",
      r.stdout + r.stderr)
r = prun(["-c", "import importlib.util as u; print(u.find_spec('torch') is None, u.find_spec('cv2') is None)"])
check("S2 neither torch nor cv2 importable from the fresh unzip", r.stdout.strip() == "True True", r.stdout + r.stderr)
r = prun(["-m", "vathography_app", "depth", os.path.join(work, "no_photo.png"), "--out", os.path.join(work, "out")])
check("S3 depth refused with a sentence, no traceback",
      r.returncode == 2 and "not installed yet" in r.stderr and "Traceback" not in r.stderr, f"rc {r.returncode} {r.stderr}")
r = prun(["-m", "vathography_app", "install-third-party", "--wheels", wheels], timeout=1800)
check("S4 PyTorch + OpenCV installed offline into Application Support",
      r.returncode == 0 and os.path.isdir(os.path.join(support, "third_party")), r.stdout + r.stderr)
r = prun(["-c", "from vathography_app import third_party as T; T.add_to_path(); import torch, cv2; "
                "print(torch.__file__); print(cv2.__file__); print(torch.backends.mps.is_built())"])
lines = r.stdout.splitlines()                             # by LINE: the paths hold "Application Support" (a space)
check("S5 torch + cv2 load from Application Support/third_party, MPS built in",
      r.returncode == 0 and len(lines) == 3 and all(l.startswith(os.path.join(support, "third_party")) for l in lines[:2])
      and lines[2] == "True", r.stdout + r.stderr)
r = prun(["-c", "import importlib, pkgutil, vathography_app as v\n"
                "from vathography_app import third_party as T; T.add_to_path()\n"
                "n = 0\n"
                "for m in pkgutil.walk_packages(v.__path__, 'vathography_app.'):\n"
                "    if m.name.endswith('__main__'): continue\n"
                "    importlib.import_module(m.name); n += 1\n"
                "print(n)"])
check("S5 every module of vathography_app imports (CPU)", r.returncode == 0, r.stdout + r.stderr)
exe = os.path.join(contents, "MacOS", NAME)
p = subprocess.Popen([exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(20)
alive = p.poll() is None
if alive:
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()
out, err = p.communicate()
check("S6 the bundle's executable starts the window (running after 20 s)", alive, f"rc {p.returncode} {out} {err}")
ok, info = verify()
check("S7 nothing written into the bundle (signature still verifies)", ok, info)
shutil.rmtree(work, ignore_errors=True)
print(f"smoke: {sum(res)}/{len(res)}", flush=True)
sys.exit(0 if all(res) else 1)
