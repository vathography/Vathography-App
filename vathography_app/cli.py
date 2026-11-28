# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""cli.py — `python -m vathography_app depth <photo or folder>... [--see-through] [--format exr32|exr16|png16|png8]
[--models DIR] [--out DIR] [--budget GIB] [--verify] [--progress-json]` — the app core without a window.
One photo at a time, each written as soon as it is done, next to the photo unless --out is given, as
<stem>_depth_v001.<ext>, _v002, ... (never over an existing file).
--progress-json: every line on stdout is one JSON event for the window's worker reader —
  {"ev": "log", "msg"} · "model" (loaded) · "photo" (i, n, path) · "step" (i, done, total) · "written" (i, files, s)
  · "error" (i, path, msg) · "done" (ok, failed) · "stage" (i, name[, done, total]) · "warning" (i, msg) · "fatal"
  (msg, trace). The maps are unchanged (the flag only prints).
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".exr")
# a relief this app wrote: `_relief_strength<N>` or the earlier `_relief_level<N>`, + `_vNNN`
RELIEF_OUT = re.compile(r"_relief_(strength|level)[1-6](_v\d{3,})?$")
# a depth map this app wrote: `_depth` / `_depth_see_through`, + `_vNNN` since 0.1.2
DEPTH_OUT = re.compile(r"_depth(_see_through)?(_v\d{3,})?$")


JSON = False


def emit(ev, **kw):
    if JSON:
        print(json.dumps({"ev": ev, **kw}), flush=True)


_last = {"key": None, "t": 0.0}


def stage(i, name, done=None, total=None):
    """{"ev": "stage", "i", "name", "done", "total"}; i = the photo (0 = before the photos: loading).
    Throttle: a stage whose words (digits aside) are the same as the last one is sent at most every 0.2 s, except
    its last event (done == total); a new stage is always sent at once."""
    if not JSON:
        return
    key = (i, re.sub(r"\d+", "#", name))
    now = time.time()
    if key == _last["key"] and now - _last["t"] < 0.2 and not (total and done == total):
        return
    _last.update(key=key, t=now)
    kw = {"i": i, "name": name}
    if total:
        kw.update(done=done, total=total)
    emit("stage", **kw)


def log(*a):
    if JSON:
        emit("log", msg=" ".join(str(x) for x in a))
    else:
        print("[vathography]", *a, flush=True)


def is_app_output(path):
    """An earlier output of this app beside the photos (a folder re-dropped after a run): never read as a photo."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return bool(DEPTH_OUT.search(stem) or RELIEF_OUT.search(stem))


def photos(args, kind="depth"):
    """Files as given; a folder -> its pictures, minus this app's own earlier outputs (depth: depth maps and reliefs,
    relief: reliefs — a depth map IS a relief input)."""
    out = []
    for p in args:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                q = os.path.join(p, f)
                if not f.lower().endswith(EXTS) or not os.path.isfile(q):
                    continue
                stem = os.path.splitext(f)[0]
                if kind == "depth" and is_app_output(q):
                    continue
                if kind == "relief" and RELIEF_OUT.search(stem):
                    continue
                out.append(q)
        else:
            out.append(p)
    return out


def depth(ns):
    stage(0, "starting the depth engine")    # before the imports below (several silent seconds)
    from . import backend, device, models, outnames, photo_io, pipeline
    backend.log = log                       # the backend's own "[backend]" lines become JSON events too
    dev = device.pick()
    log(device.system_line(dev))            # in every run log: what a report needs (OS, GPU, memory, torch)
    model = "see-through" if ns.see_through else "depth"
    stage(0, "checking the model files")
    hashed = [0]

    def hashing(n):                         # only when a file has to be read (verify, or another file name)
        hashed[0] += n
        stage(0, f"checking the model files · {hashed[0] / 1e9:.1f} GB read")
    paths = models.locate(ns.models, model, verify=ns.verify, progress=hashing)
    log(f"model {model}: " + " | ".join(f"{k} {os.path.basename(v)}" for k, v in paths.items()))
    m = backend.Model(paths["backbone"], paths["lora"], paths["vae"], paths["conditioning"], models.VAE_CONFIG_DIR,
                      device=dev, budget_gib=ns.budget, log_fn=log, stage=lambda name, d=None, t=None: stage(0, name, d, t), label=model)
    assert m.recipe == model, (m.recipe, model)
    # writer kind per output: exr32 / exr16 / png16 / png8 (the window's list); png / exr / both = the
    # earlier names (16-bit PNG / 32-bit EXR / both), still read from settings saved by earlier versions
    kinds = {"exr32": ["exr"], "exr16": ["exr16"], "png16": ["png"], "png8": ["png8"],
             "png": ["png"], "exr": ["exr"], "both": ["png", "exr"]}[ns.format]
    suffix = "_depth_see_through" if ns.see_through else "_depth"
    emit("model", model=model, recipe=m.recipe, group=m.group)
    todo = photos(ns.photos)
    failed = 0
    peak = device.Peak(dev)

    def staged(i, name, d=None, t=None):    # every progress event also samples the Mac's GPU memory (no peak counter)
        peak.poll()
        stage(i, name, d, t)
    for i, p in enumerate(todo, 1):
        emit("photo", i=i, n=len(todo), path=p)
        try:
            t0 = time.time()
            peak.reset()
            stage(i, "reading the photo")
            bgr = photo_io.read_photo_bgr_u8(p)
            out, info = pipeline.run_pipeline(bgr, m.infer, pipeline.TILE, log=log,
                                              progress=(lambda d, t, i=i: (peak.poll(), emit("step", i=i, done=d, total=t))) if JSON else None,
                                              merge_device=m.dev, recipe=m.recipe,
                                              stage=(lambda name, d=None, t=None, i=i: staged(i, name, d, t)) if JSON else None,
                                              sky=True, sky_chunk=m.group,
                                              warn=lambda msg, i=i, p=p: (emit("warning", i=i, msg=f"{os.path.basename(p)}: {msg}") if JSON else log(f"{os.path.basename(p)}: {msg}")))
            stem = os.path.splitext(os.path.basename(p))[0]
            d = ns.out or os.path.dirname(os.path.abspath(p))
            os.makedirs(d, exist_ok=True)
            nxt = max(outnames.highest_version(d, f"{stem}{suffix}") + 1, 1)
            stage(i, f"writing {stem}{suffix}_v{nxt:03d}.{kinds[-1][:3]}")
            # never over an existing file: <stem><suffix>_vNNN.<ext>, one number for the whole set
            files = outnames.write_versioned(d, f"{stem}{suffix}", [(k[:3], photo_io.encode_depth(out, k)) for k in kinds])
            log(f"[{i}] {stem}: {info['mode']} | {time.time() - t0:.0f} s | {peak.line()} -> " + " | ".join(files))
            emit("written", i=i, files=files, s=round(time.time() - t0, 1))
        except Exception as e:
            if not JSON:
                raise
            failed += 1
            emit("error", i=i, path=p, msg=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        device.empty_cache(dev)
    emit("done", ok=len(todo) - failed, failed=failed)
    return 0


# ----------------------------------------------------------------------------- Mac: automatic follow-up
# On a Mac the window's worker runs as a SUPERVISOR (no torch): the photos run in a child process (attempt 1). If a
# photo stops with "MPS backend out of memory", the supervisor runs, each in a fresh process (all of the GPU memory
# back): attempt 2 = a memory probe of the model's phases on that photo (writes no map), attempt 3 = the failed photos
# again with 1 attention head per call (exact per head). Children inherit stderr => everything lands in ONE
# last_run.log (the window opens it anew per launch, in gui.py). Windows: never used.

ATTEMPTS = ("run", "probe", "heads1")
CHILD_CMD = None                             # set by the Mac build's checks: a stand-in child command


def _watch_parent():
    """A child stops when its supervisor is gone (the window's Cancel kills the supervisor with SIGKILL)."""
    import threading
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


MPS_OOM_TEXT = "MPS backend out of memory"   # = device.MPS_OOM_TEXT (device imports torch; the supervisor must not)


class device_text:
    @staticmethod
    def is_mps_oom(msg):
        return MPS_OOM_TEXT in msg


def supervise(argv, ns):
    """-> exit code. argv = this worker's own arguments ("depth", photos..., options)."""
    import subprocess
    todo = photos(ns.photos)
    n_pos = len(ns.photos)
    opts = argv[1 + n_pos:]
    base = CHILD_CMD or [sys.executable, "-E", "-s", "-u", "-m", "vathography_app"]

    def child(paths, attempt, extra, on_event):
        cmd = base + ["depth"] + list(paths) + opts + ["--attempt", attempt] + extra
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                ev = {"ev": "log", "msg": line}
            on_event(ev)
        return p.wait()

    ok, failed, oom = [0], [0], []

    def first(ev):
        k = ev.get("ev")
        if k == "error" and device_text.is_mps_oom(ev.get("msg", "")):
            oom.append(ev["i"])
            emit("log", msg=f"attempt 1 of 3: {os.path.basename(ev['path'])}: {ev['msg']} -- the app tries again "
                            f"by itself (a memory check, then 1 attention head per call)")
            log(ev.get("trace", ""))
            return
        if k == "fatal" and device_text.is_mps_oom(ev.get("msg", "")):
            oom.extend(range(1, len(todo) + 1))
            emit("log", msg=f"attempt 1 of 3: {ev['msg']} -- the app tries again by itself")
            return
        if k == "done":
            return
        if k == "written":
            ok[0] += 1
        elif k == "error":
            failed[0] += 1
        print(json.dumps(ev), flush=True)
    log("attempt 1 of 3: the photos")
    code = child(todo, "run", [], first)
    if not oom:
        emit("done", ok=ok[0], failed=failed[0])
        return code
    redo = [todo[i - 1] for i in oom]
    stage(oom[0], "checking the graphics memory (automatic second try)")
    log(f"attempt 2 of 3: memory probe on {os.path.basename(redo[0])} (no map is written)")

    def second(ev):
        if ev.get("ev") in ("log", "fatal", "error"):
            emit("log", msg="[probe] " + (ev.get("msg") or ""))
    child(redo[:1], "probe", [], second)
    log(f"attempt 3 of 3: {len(redo)} photo(s) again, 1 attention head per call")

    def third(ev):
        k = ev.get("ev")
        if k == "done":
            return
        if "i" in ev and isinstance(ev["i"], int) and ev["i"] >= 1:
            ev = dict(ev, i=oom[ev["i"] - 1])
        if k == "photo":
            ev["n"] = len(todo)
        if k == "written":
            ok[0] += 1
        elif k == "error":
            failed[0] += 1
        elif k == "fatal":                              # nothing ran: every photo of this attempt failed
            failed[0] += len(redo)
            for i, p in zip(oom, redo):
                emit("error", i=i, path=p, msg=ev.get("msg", ""), trace=ev.get("trace", ""))
            return
        print(json.dumps(ev), flush=True)
    child(redo, "heads1", ["--mps-heads", "1"], third)
    emit("done", ok=ok[0], failed=failed[0])
    return 0


def probe(ns):
    """Attempt 2 (a child, Mac): the model's phases on the photo, one memory line after each; writes no map."""
    import torch
    from . import backend, device, models, photo_io
    from .engine import tiled_core as TC
    dev = device.pick()
    log(device.system_line(dev))
    model = "see-through" if ns.see_through else "depth"
    paths = models.locate(ns.models, model, verify=False)
    m = backend.Model(paths["backbone"], paths["lora"], paths["vae"], paths["conditioning"], models.VAE_CONFIG_DIR,
                      device=dev, budget_gib=ns.budget, log_fn=log, label=model)
    p = photos(ns.photos)[0]
    bgr = photo_io.read_photo_bgr_u8(p)
    H, W = bgr.shape[:2]
    R = TC.RECIPES[m.recipe]
    mw, mh, mp = TC.pin_for16(W, H, TC.M1_BUDGET_MP, R["m1_strip"], R["m1_max_long_edge"])
    decoy = TC.load_decoy(R["m1_decoy_scale"]) if R["m1_decoy_scale"] > 0 else None
    x = TC.build_m1_input(bgr, mw, mh, decoy, R["m1_strip"])
    log(f"probe: global input {x.shape[1]}x{x.shape[0]} ({mp:.2f} MP)")
    m.mem("probe: photo read")

    def step(label, fn):
        """fn() on the GPU; an out-of-memory is reported and the probe goes on. The reading is taken outside the
        except block (inside it the traceback holds the failed call's tensors)."""
        msg = None
        try:
            out = fn()
        except RuntimeError as e:
            if not device.is_mps_oom(e):
                raise
            msg, out = str(e).split(". Tried")[0], None
        m.mem(f"probe: {label}" + (f" -- OUT OF MEMORY ({msg})" if msg else ""))
        return out
    lat = step(f"after the encode, image decoder on the {'processor' if m.vae_cpu else 'GPU'}",
               lambda: backend.encode(m.vae, m.mean, m.std, x).cpu())
    if lat is not None:
        shim = type("OneBlock", (), {})()
        for n in ("img_in", "txt_norm", "txt_in", "time_text_embed", "pos_embed", "norm_out", "proj_out"):
            setattr(shim, n, getattr(m.tr, n))
        shim.transformer_blocks = m.tr.transformer_blocks[:1]
        with device.attention(dev):
            step("after block 1 (the transformer's first block only)",
                 lambda: backend.step_layer_major(shim, m.streamer, m.cond, [lat], dev, mem=m.mem))
        m.release_all()
        m.mem("probe: block 1 released")
    if m.vae_cpu:
        m.vae.to(dev, torch.bfloat16)                   # the image decoder on the GPU
        m.mem("probe: image decoder moved to the GPU (bfloat16)")
        lat2 = step("after the encode, image decoder on the GPU",
                    lambda: backend.encode(m.vae, m.mean, m.std, x))
        if lat2 is not None:
            step("after the decode, image decoder on the GPU",
                 lambda: backend.decode_raw(m.vae, m.mean, m.std, lat2))
        del lat2
        m.vae.to("cpu")
        m.mem("probe: image decoder back on the processor")
    log("probe: done")
    return 0


def relief(ns):
    stage(0, "starting the depth engine")    # relief_app imports torch at its top
    from . import relief_app
    todo = photos(ns.files, kind="relief")
    failed = 0
    for i, p in enumerate(todo, 1):
        emit("photo", i=i, n=len(todo), path=p)
        try:
            t0 = time.time()
            dst, info = relief_app.run(p, ns.strength, budget_gib=ns.budget, out_dir=ns.out, log=log,
                                       stage=(lambda name, d=None, t=None, i=i: stage(i, name, d, t)) if JSON else None)
            log(f"{os.path.basename(p)} -> {dst} | {time.time() - t0:.1f} s")
            emit("written", i=i, files=[dst], s=round(time.time() - t0, 1))
        except Exception as e:
            if not JSON:
                raise
            failed += 1
            emit("error", i=i, path=p, msg=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    emit("done", ok=len(todo) - failed, failed=failed)
    return 0


def install_third_party(argv):
    """`install-third-party --wheels DIR`: the same install() the window runs, from the pinned wheels found in DIR
    (never deleted). Not offered in the window."""
    from . import third_party
    ap = argparse.ArgumentParser(prog="vathography_app install-third-party")
    ap.add_argument("--wheels", required=True)
    ns = ap.parse_args(argv)
    for e in third_party.entries():
        whl = os.path.join(ns.wheels, e["file"])
        if third_party.installed(e):
            print(f"[vathography] {e['name']} {e['version']}: already installed", flush=True)
            continue
        if not os.path.isfile(whl):
            print(f"[vathography] {e['name']}: {e['file']} not found in {ns.wheels}", file=sys.stderr, flush=True)
            return 2
        p = third_party.install(e, whl)
        print(f"[vathography] {e['name']} {e['version']} installed in {p}", flush=True)
    return 0


def main(argv=None):
    from . import platform_io
    if platform_io.IS_MAC:
        # an operation Metal does not have runs on the CPU instead of stopping the photo (PyTorch logs each one)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # memory log: PyTorch prints its own split ("MPS allocated" / "other allocations") at each cache
        # release (device.mem_point) to stderr = last_run.log. Read once by PyTorch, so set before torch is imported.
        os.environ.setdefault("PYTORCH_DEBUG_MPS_ALLOCATOR", "1")
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["gui"]:                 # the window; files after `gui` are listed in the Depth tab
        from . import gui
        return gui.main(argv[1:])
    if argv[:1] == ["install-third-party"]:  # hidden: install from wheels already on disk, offline (the Mac build's checks)
        return install_third_party(argv[1:])
    from . import models                    # light (no torch): the default models folder
    ap = argparse.ArgumentParser(prog="vathography_app")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("depth", help="depth maps from photos")
    d.add_argument("photos", nargs="+")
    d.add_argument("--see-through", action="store_true", help="the see-through model (depth behind glass)")
    d.add_argument("--format", choices=["exr32", "exr16", "png16", "png8", "png", "exr", "both"], default="png",
                   help="exr32 / exr16 = 32-bit / 16-bit (half) float EXR · png16 / png8 = 16-bit / 8-bit PNG · "
                        "png = png16, exr = exr32, both = png16 + exr32")
    d.add_argument("--models", default=models.DEFAULT_DIR, help="folder holding the model files (default: the "
                                                                    "models folder beside the app)")
    d.add_argument("--out", default=None)
    d.add_argument("--budget", type=float, default=6.5, help="GPU memory the app may use, GiB")
    d.add_argument("--verify", action="store_true", help="check every model file's sha256 first")
    d.add_argument("--progress-json", action="store_true", help="one JSON event per line on stdout (the window's worker)")
    d.add_argument("--attempt", choices=ATTEMPTS, default=None, help=argparse.SUPPRESS)   # Mac supervisor's children
    d.add_argument("--mps-heads", type=int, default=None, help=argparse.SUPPRESS)        # attempt 3: heads per call
    r = sub.add_parser("relief", help="relief increase on an add-on export or an app depth map")
    r.add_argument("files", nargs="+")
    r.add_argument("--strength", "--level", dest="strength", type=int, choices=range(1, 7), default=1,
                   help="relief strength 1-6 (--level = the earlier name)")
    r.add_argument("--out", default=None)
    r.add_argument("--budget", type=float, default=6.5)
    r.add_argument("--progress-json", action="store_true", help="one JSON event per line on stdout (the window's worker)")
    ns = ap.parse_args(argv)
    global JSON
    JSON = ns.progress_json
    if (ns.cmd == "depth" and platform_io.IS_MAC and JSON and ns.attempt is None
            and argv[1:1 + len(ns.photos)] == ns.photos):           # the window's form: photos right after "depth"
        return supervise(argv, ns)
    if ns.cmd == "depth" and ns.attempt:
        _watch_parent()
    from . import third_party              # stdlib only: PyTorch + OpenCV present and loading, else one sentence
    third_party.add_to_path()               # macOS: third_party's folders on sys.path (Windows: the package's .pth)
    miss = third_party.missing()
    if not miss:
        stage(0, "starting the depth engine")   # the load check imports torch: several silent seconds otherwise
    problem = third_party.not_ready_sentence(miss) if miss else third_party.load_check()
    if problem:
        if JSON:
            emit("fatal", msg=problem, trace="")
        else:
            print("[vathography]", problem, file=sys.stderr, flush=True)
        return 2
    try:
        if ns.cmd == "depth" and ns.mps_heads:
            from . import device
            device.MPS_HEADS_PER_CALL = ns.mps_heads
            log(f"attention: {ns.mps_heads} head(s) per call on the Mac's GPU")
        if ns.cmd == "depth" and ns.attempt in ("probe", "heads1"):
            from . import backend               # after an out-of-memory: the decode stays on the processor
            backend.DECODE_ON_GPU_MAC = False
            log("image decoder: on the processor for this attempt")
        if ns.cmd == "depth" and ns.attempt == "probe":
            return probe(ns)
        return depth(ns) if ns.cmd == "depth" else relief(ns)
    except Exception as e:                  # before / outside the per-photo loop: models missing, no GPU, ...
        if not JSON:
            raise
        emit("fatal", msg=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
