# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""cli.py — `python -m vathography_app depth <photo or folder>... [--see-through] [--format exr32|exr16|png16|png8]
[--models DIR] [--out DIR] [--budget GIB] [--verify] [--progress-json]` — the app core without a window.
One photo at a time, each written as soon as it is done, next to the photo unless --out is given, as
<stem>_depth_v001.<ext>, _v002, ... (never over an existing file).
--progress-json: every line on stdout is one JSON event for the window's worker reader —
  {"ev": "log", "msg"} · "model" (loaded) · "photo" (i, n, path) · "step" (i, done, total) · "written" (i, files, s)
  · "error" (i, path, msg) · "done" (ok, failed). The maps are unchanged (the flag only prints).
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".exr")
# a relief this app wrote: `_relief_strength<N>` (since 2026-10-31) or the earlier `_relief_level<N>`, + `_vNNN` (0.1.2)
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
    import torch
    from . import backend, models, outnames, photo_io, pipeline
    backend.log = log                       # the backend's own "[backend]" lines become JSON events too
    if ns.cap:                              # test option: torch's allocator hard-capped (an 8 GB card ~ 6.5 GiB)
        dev0 = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(ns.cap * 2 ** 30 / torch.cuda.get_device_properties(dev0).total_memory, dev0)
        log(f"GPU memory capped at {ns.cap} GiB for torch")
    model = "see-through" if ns.see_through else "depth"
    stage(0, "checking the model files")
    hashed = [0]

    def hashing(n):                         # only when a file has to be read (verify, or another file name)
        hashed[0] += n
        stage(0, f"checking the model files · {hashed[0] / 1e9:.1f} GB read")
    paths = models.locate(ns.models, model, verify=ns.verify, progress=hashing)
    log(f"model {model}: " + " | ".join(f"{k} {os.path.basename(v)}" for k, v in paths.items()))
    m = backend.Model(paths["backbone"], paths["lora"], paths["vae"], paths["conditioning"], models.VAE_CONFIG_DIR,
                      budget_gib=ns.budget, log_fn=log, stage=lambda name, d=None, t=None: stage(0, name, d, t), label=model)
    assert m.recipe == model, (m.recipe, model)
    # writer kind per output: exr32 / exr16 / png16 / png8 (the window's list, 2026-10-31); png / exr / both = the
    # earlier names (16-bit PNG / 32-bit EXR / both), kept for the gates and old settings
    kinds = {"exr32": ["exr"], "exr16": ["exr16"], "png16": ["png"], "png8": ["png8"],
             "png": ["png"], "exr": ["exr"], "both": ["png", "exr"]}[ns.format]
    suffix = "_depth_see_through" if ns.see_through else "_depth"
    emit("model", model=model, recipe=m.recipe, group=m.group)
    todo = photos(ns.photos)
    failed = 0
    for i, p in enumerate(todo, 1):
        emit("photo", i=i, n=len(todo), path=p)
        try:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            stage(i, "reading the photo")
            bgr = photo_io.read_photo_bgr_u8(p)
            out, info = pipeline.run_pipeline(bgr, m.infer, pipeline.TILE, log=log,
                                              progress=(lambda d, t, i=i: emit("step", i=i, done=d, total=t)) if JSON else None,
                                              merge_device=m.dev, recipe=m.recipe,
                                              stage=(lambda name, d=None, t=None, i=i: stage(i, name, d, t)) if JSON else None,
                                              sky=not ns.no_sky_repair, sky_chunk=m.group, sky_limit=ns.sky_frames, sky_scaled=not ns.no_scaled_frames,
                                              warn=lambda msg, i=i, p=p: (emit("warning", i=i, msg=f"{os.path.basename(p)}: {msg}") if JSON else log(f"{os.path.basename(p)}: {msg}")))
            stem = os.path.splitext(os.path.basename(p))[0]
            d = ns.out or os.path.dirname(os.path.abspath(p))
            os.makedirs(d, exist_ok=True)
            nxt = max(outnames.highest_version(d, f"{stem}{suffix}") + 1, 1)
            stage(i, f"writing {stem}{suffix}_v{nxt:03d}.{kinds[-1][:3]}")
            # never over an existing file: <stem><suffix>_vNNN.<ext>, one number for the whole set
            files = outnames.write_versioned(d, f"{stem}{suffix}", [(k[:3], photo_io.encode_depth(out, k)) for k in kinds])
            log(f"[{i}] {stem}: {info['mode']} | {time.time() - t0:.0f} s | GPU peak allocated "
                f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} / reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB -> "
                + " | ".join(files))
            emit("written", i=i, files=files, s=round(time.time() - t0, 1))
        except Exception as e:
            if not JSON:
                raise
            failed += 1
            emit("error", i=i, path=p, msg=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        torch.cuda.empty_cache()
    emit("done", ok=len(todo) - failed, failed=failed)
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
    # before torch is imported. ⚠ torch on Windows does NOT support expandable segments (it warns and ignores the
    # setting) — kept for other platforms.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["gui"]:                 # the window; files after `gui` are listed in the Depth tab
        from . import gui
        return gui.main(argv[1:])
    if argv[:1] == ["install-third-party"]:  # hidden, for the gates: install from wheels already on disk, offline
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
    d.add_argument("--cap", type=float, default=None, help="test option: hard cap on torch's GPU memory, GiB")
    d.add_argument("--no-sky-repair", action="store_true", help="test option: skip the sky repair")
    d.add_argument("--no-scaled-frames", action="store_true", help="test option: the sky repair without its scaled frames")
    d.add_argument("--sky-frames", type=int, default=None, help="test option: most full frames on one photo, one tile's frames at the least; scaled frames are not counted")
    d.add_argument("--progress-json", action="store_true", help="one JSON event per line on stdout (the window's worker)")
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
    from . import third_party               # stdlib only: PyTorch + OpenCV present and loading, else one sentence
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
        return depth(ns) if ns.cmd == "depth" else relief(ns)
    except Exception as e:                  # before / outside the per-photo loop: models missing, no GPU, ...
        if not JSON:
            raise
        emit("fatal", msg=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
