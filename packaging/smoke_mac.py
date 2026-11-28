r"""smoke_mac.py — checks on the BUILT macOS zip, on the GitHub Actions runner (no GPU there: no depth map is made).
What a runner can check on the built package:
  S1 the zip unpacks (ditto) and the bundle's signature verifies
  S2 the window's code imports with neither torch nor cv2; neither is importable from a fresh unpack
  S3 `depth` is refused with one sentence, no traceback
  S4 PyTorch + OpenCV installed OFFLINE from the wheels the build resolved, through the hidden `install-third-party`
     verb (the window's install()), into ~/Library/Application Support/Vathography App/third_party
  S5 after S4: torch + cv2 load from that folder; torch has MPS built in; every module of vathography_app imports (CPU)
  S6 the bundle's own executable starts the window: still running after 20 s, printed "[gui] window ready", no
     traceback (then closed)
  C6 the automatic follow-up after a Mac out-of-memory (cli.supervise), with the bundle's own Python:
     C6a the window's exact command, a real child, no model files: the child's failure reaches the window, no follow-up
     C6b a stand-in child that hits a REAL "MPS backend out of memory" (tiny memory cap): the memory check (attempt 2)
         and the 1-head re-run (attempt 3) follow by themselves, the photo is written, every child's stderr in one log
     C6c the window's Cancel (SIGKILL on the supervisor): its child ends by itself within a few seconds
     C6d the guarded decode: a REAL out-of-memory in the image decoder's GPU copy -> that decode and the
         next one on the processor, maps finite (random weights from the bundled config)
  S7 nothing was written into the bundle by S2-C6 (its signature still verifies)
Usage (macOS): python3 packaging/smoke_mac.py <zip> <wheels folder>
"""
import json
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
if os.environ.get("GITHUB_ACTIONS") == "true":
    shutil.rmtree(support, ignore_errors=True)           # a CI runner: the app's data folder starts empty
elif os.path.exists(support):
    sys.exit(f"{support} exists: this check needs a Mac without the app's data (models, settings); "
             "it runs on GitHub Actions")
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
if os.environ.get("SEED_DEFECT") == "window":            # must-fire test run: S6 has to go red on a failing window
    gpy = os.path.join(appdir, "vathography_app", "gui.py")
    src = open(gpy, encoding="utf-8").read()
    assert src.count("    root.mainloop()") == 1, "seed window: root.mainloop() not found once"
    open(gpy, "w", encoding="utf-8").write(src.replace("    root.mainloop()", "    root.after(500, lambda: 1 / 0)\n    root.mainloop()"))
    pyc = os.path.join(appdir, "vathography_app", "__pycache__")
    for n in os.listdir(pyc):
        if n.startswith("gui."):
            os.remove(os.path.join(pyc, n))
    print("SEEDED DEFECT window: the unzipped gui.py raises in a Tk callback 0.5 s after start (the process stays "
          "alive); S6 and S7 'signature' must FAIL", flush=True)


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
# alive alone cannot tell the window from a window whose code failed (a Tk callback error keeps the process running):
# the window prints "[gui] window ready" once built (gui._say_ready) and no traceback may appear (seed "window")
check("S6 the bundle's executable starts the window (running after 20 s, 'window ready', no traceback)",
      alive and "[gui] window ready" in err and "Traceback" not in err, f"rc {p.returncode} {out} {err}")

# C6 -------------------------------------------------------------------------------------------- Mac supervisor
sup = os.path.join(work, "c6")
os.makedirs(sup)
ENV0 = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
SEED_C6 = os.environ.get("SEED_DEFECT") == "supervisor"   # must-fire test run: C6b has to go red
if SEED_C6:
    print("SEEDED DEFECT supervisor: the stand-in child's error carries another text, not the out-of-memory one; "
          "C6b must FAIL", flush=True)


def events(text):
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def window_form(args, logname, env=None, timeout=600):
    """The window's worker command (gui.Worker): -E -s -u -m vathography_app ... --progress-json, cwd = the app
    folder, stderr = one log file. -> (returncode, events, log text)"""
    logf = os.path.join(sup, logname)
    with open(logf, "w", encoding="utf-8") as lf:
        r = subprocess.run([py, "-E", "-s", "-u", "-m", "vathography_app"] + args + ["--progress-json"], cwd=appdir,
                           env=env or ENV0, stdout=subprocess.PIPE, stderr=lf, text=True, timeout=timeout)
    return r.returncode, events(r.stdout), open(logf, encoding="utf-8").read()


photo = os.path.join(sup, "grey.png")
r = prun(["-c", "from vathography_app import third_party as T; T.add_to_path(); import cv2, numpy as np; "
                f"print(cv2.imwrite({photo!r}, np.full((64, 96, 3), 128, np.uint8)))"])
nomodels = os.path.join(sup, "no_models")
os.makedirs(nomodels)
code, evs, logtxt = window_form(["depth", photo, "--format", "exr32", "--out", os.path.join(sup, "out_a"),
                                 "--models", nomodels], "last_run_a.log")
bad = [e for e in evs if e.get("ev") in ("fatal", "error")]
msgs = " | ".join(e.get("msg", "") for e in evs if e.get("ev") == "log")
check("C6a real child from the bundle: its failure (no model files) reaches the window, no follow-up",
      r.stdout.strip() == "True" and bad and not any("MPS backend out of memory" in e.get("msg", "") for e in bad)
      and "attempt 1 of 3" in msgs and "attempt 2 of 3" not in msgs and "attempt 3 of 3" not in msgs
      and [e for e in evs if e.get("ev") == "done"] != [], f"rc {code} events {evs} log {logtxt[-600:]}")

# C6b: the supervisor driven with a stand-in child (CHILD_CMD); attempt 1 of
# the stand-in fills the GPU under a tiny cap until PyTorch raises its real out-of-memory error, and reports it as the
# real child does (an "error" event with str(e))
FAKE = r'''
import json, os, sys
sys.path.insert(0, os.environ["C6_APPDIR"])
a = sys.argv[1:]
att = a[a.index("--attempt") + 1]
photos = [x for x in a[1:] if x.endswith(".png")]
def ev(**k): print(json.dumps(k), flush=True)
print(f"stderr of the {att} child", file=sys.stderr, flush=True)
if att == "run":
    from vathography_app import third_party as T
    T.add_to_path()
    import torch
    ev(ev="model", model="depth", recipe="depth", group=23)
    ev(ev="photo", i=1, n=1, path=photos[0])
    held, msg = [], "no error"
    try:
        for _ in range(4096):
            held.append(torch.empty(64 * 2 ** 20, dtype=torch.uint8, device="mps"))
    except RuntimeError as e:
        msg = str(e)
    del held
    if os.environ.get("SEED_DEFECT") == "supervisor":
        msg = "RuntimeError: some other failure"
    ev(ev="error", i=1, path=photos[0], msg=msg, trace="")
    ev(ev="done", ok=0, failed=1)
elif att == "probe":
    ev(ev="log", msg="memory probe ran")
elif att == "heads1":
    ev(ev="log", msg="heads " + a[a.index("--mps-heads") + 1])
    ev(ev="model", model="depth", recipe="depth", group=23)
    ev(ev="photo", i=1, n=1, path=photos[0])
    ev(ev="written", i=1, files=[photos[0] + ".exr"], s=1.0)
    ev(ev="done", ok=1, failed=0)
else:                                    # "watch": C6c
    import vathography_app.cli as C
    C._watch_parent()
    ev(ev="log", msg="pid " + str(os.getpid()))
    import time
    time.sleep(120)
'''
DRIVER = r'''
import json, os, sys
sys.path.insert(0, os.environ["C6_APPDIR"])
import vathography_app.cli as C
C.CHILD_CMD = [sys.executable, "-E", "-s", "-u", sys.argv[1]]
sys.exit(C.main(json.loads(sys.argv[2])))
'''
fake, drv = os.path.join(sup, "stand_in_child.py"), os.path.join(sup, "driver.py")
open(fake, "w", encoding="utf-8").write(FAKE)
open(drv, "w", encoding="utf-8").write(DRIVER)
# the low ratio must not exceed the high one (PyTorch refuses: "invalid low watermark ratio")
env_b = dict(ENV0, C6_APPDIR=appdir, PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.05",
             PYTORCH_MPS_LOW_WATERMARK_RATIO="0.04")
logf = os.path.join(sup, "last_run_b.log")
with open(logf, "w", encoding="utf-8") as lf:
    r = subprocess.run([py, "-E", "-s", "-u", drv, fake, json.dumps(["depth", photo, "--format", "exr32",
                        "--progress-json"])], cwd=appdir, env=env_b, stdout=subprocess.PIPE, stderr=lf, text=True,
                       timeout=600)
evs, logtxt = events(r.stdout), open(logf, encoding="utf-8").read()
msgs = [e.get("msg", "") for e in evs if e.get("ev") == "log"]
check("C6b a real MPS out-of-memory: attempt 1's error shown as a log line, not as the photo's error",
      any(m.startswith("attempt 1 of 3:") and "MPS backend out of memory" in m for m in msgs)
      and not [e for e in evs if e.get("ev") == "error"], f"rc {r.returncode} events {evs}")
check("C6b the memory check (attempt 2) then the 1-head re-run (attempt 3) followed by themselves",
      any(m == "[probe] memory probe ran" for m in msgs) and "heads 1" in msgs
      and any(e.get("ev") == "written" and e.get("i") == 1 for e in evs)
      and [e for e in evs if e.get("ev") == "done"] == [{"ev": "done", "ok": 1, "failed": 0}], f"events {evs}")
check("C6b one log: every child's stderr",
      all(f"stderr of the {a} child" in logtxt for a in ("run", "probe", "heads1")), logtxt[-600:])

# C6c: the window's Cancel kills the supervisor with SIGKILL; the child (any attempt) must end by itself
p = subprocess.Popen([py, "-E", "-s", "-u", "-c",
                      "import os, sys, subprocess; sys.exit(subprocess.call([sys.executable, '-E', '-s', '-u', "
                      f"{fake!r}, 'depth', {photo!r}, '--attempt', 'watch']))"],
                     cwd=appdir, env=dict(ENV0, C6_APPDIR=appdir), stdout=subprocess.PIPE, text=True)
child_pid = None
t0 = time.time()
while time.time() - t0 < 120:
    line = p.stdout.readline()
    if not line:
        break
    for e in events(line):
        if e.get("msg", "").startswith("pid "):
            child_pid = int(e["msg"][4:])
    if child_pid:
        break
p.send_signal(signal.SIGKILL)
p.wait()


def alive_pid(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


gone = False
if child_pid:
    t0 = time.time()
    while time.time() - t0 < 10:
        if not alive_pid(child_pid):
            gone = True
            break
        time.sleep(0.25)
    if not gone:
        os.kill(child_pid, signal.SIGKILL)
check("C6c the window's Cancel (SIGKILL on the supervisor): its child ends by itself within 10 s",
      child_pid is not None and gone, f"child pid {child_pid}")

# C6d: the guarded decode (backend.GuardedDecoder): the image decoder's GPU copy meets a REAL out-of-memory
# (Metal capped just above its weights) -> that decode and the next one run on the processor. Random weights from
# the bundled config (no model files on the runner); small latents (the processor's float32 decode is slow on ARM).
SEED_C6D = os.environ.get("SEED_DEFECT") == "decode"      # must-fire test run: C6d has to go red
if SEED_C6D:
    print("SEEDED DEFECT decode: the GPU decode's error is not caught (no fallback); C6d must FAIL", flush=True)
DEC = r'''
import copy, json, os, sys
sys.path.insert(0, os.environ["C6_APPDIR"])
from vathography_app import third_party as T
T.add_to_path()
import numpy as np, torch
from diffusers import AutoencoderKLQwenImage
from vathography_app import backend as B, models as M
if os.environ.get("SEED_DEFECT") == "decode":
    B._decode_try = lambda vae, mean, std, lat: (B.decode_raw(vae, mean, std, lat), None)
dev = torch.device("mps")
torch.manual_seed(0)
vae = AutoencoderKLQwenImage.from_config(AutoencoderKLQwenImage.load_config(M.VAE_CONFIG_DIR))
vae = vae.float().eval().requires_grad_(False)
vae.enable_tiling(tile_sample_min_height=B.VAE_TILE, tile_sample_min_width=B.VAE_TILE,
                  tile_sample_stride_height=B.VAE_STRIDE, tile_sample_stride_width=B.VAE_STRIDE)
mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1)
std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1)
gpu = copy.deepcopy(vae).to(dev, torch.bfloat16)
torch.mps.synchronize()
frac = (torch.mps.driver_allocated_memory() + 32 * 2 ** 20) / torch.mps.recommended_max_memory()
torch.mps.set_per_process_memory_fraction(frac)
lines = []
d = B.GuardedDecoder(vae, gpu, mean, std, dev, lines.append, lambda label: lines.append("memory " + label))
a = d(torch.randn(1, 16, 64, 64))
b = d(torch.randn(1, 16, 16, 16))
print(json.dumps({"lines": lines, "fallback": d.fallback, "gpu_dropped": d.vae_gpu is None, "frac": frac,
                  "shapes": [list(a.shape), list(b.shape)],
                  "finite": bool(np.isfinite(a).all() and np.isfinite(b).all())}), flush=True)
'''
dec = os.path.join(sup, "guarded_decode.py")
open(dec, "w", encoding="utf-8").write(DEC)
r = subprocess.run([py, "-E", "-s", "-u", dec], cwd=appdir, env=dict(ENV0, C6_APPDIR=appdir), stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True, timeout=1500)
res_d = (events(r.stdout) or [{}])[-1]
ln = res_d.get("lines", [])
check("C6d a real out-of-memory in the GPU decode: decoded on the processor, the next decode too, maps finite",
      r.returncode == 0 and res_d.get("gpu_dropped") is True and "out of memory" in (res_d.get("fallback") or "")
      and sum(x.startswith("decode on the GPU failed") for x in ln) == 1
      and sum(x.startswith("decode on the processor") for x in ln) == 2
      and res_d.get("shapes") == [[512, 512], [128, 128]] and res_d.get("finite") is True,
      f"rc {r.returncode} result {res_d} stderr {r.stderr[-1200:]}")
shutil.rmtree(sup, ignore_errors=True)

ok, info = verify()
check("S7 nothing written into the bundle (signature still verifies)", ok, info)
shutil.rmtree(work, ignore_errors=True)
print(f"smoke: {sum(res)}/{len(res)}", flush=True)
sys.exit(0 if all(res) else 1)
