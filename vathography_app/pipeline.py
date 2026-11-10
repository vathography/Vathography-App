# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""pipeline.py — the depth pipeline of node 1.3.0 without ComfyUI: `run_pipeline` is copied from the pinned
1.3.0 nodes.py (lines 23-92), so the app runs the node's orchestration on the node's engine (vathography_app/engine =
byte copies; same inputs -> the same map, 0.0). Two app-only departures: GLOBAL_COVER_MIN (both models, photos the
global pass already covers ~1:1 skip the tiles) and the sky repair (sky=True).
"""
import time
import traceback

from .engine import tiled_core as TC

TILE = 1536                    # = nodes.TILE (1.3.0: the only tile size; smaller tiles were judged clearly worse)

# ⚠ APP-ONLY departure from node 1.3.0: with EITHER model, no tiles while the global pass already sees the photo at
# >= GLOBAL_COVER_MIN of its size (linear: sqrt(global px / photo px)).
# Judged by eye: depth at 0.868 (1600 x 2070) global only = similar to 4 tiles; at 0.694 (2000 x 2588) the tiles add
# detail. See-through at 0.80 (1600 x 2400) similar. 0.865 = the judged depth point, rounded DOWN.
GLOBAL_COVER_MIN = 0.865


def _sky(S, G, plan, tp, rows, photo, infer, merge_device, stage, log, chunk, limit, warn, info, scaled=True):
    stage("checking the sky")
    try:
        from . import sky_repair
        res = sky_repair.repair(S, G, plan, tp, rows, photo, infer, merge_device, stage, log, chunk=chunk, limit=limit, scaled=scaled)
        info["sky"] = res[1] if res is not None else dict(flagged=[])
        return res[0] if res is not None else S
    except Exception as e:                           # a GPU out-of-memory lands here too
        import torch
        log(f"[sky_repair] FAILED, the photo keeps the unrepaired map: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        info["sky"] = dict(failed=f"{type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        if warn:
            warn(f"The sky repair stopped ({type(e).__name__}); this photo keeps its depth map without it.")
        return S


def run_pipeline(photo, infer, tile=TILE, log=print, progress=None, merge_device="cpu", m1_budget=TC.M1_BUDGET_MP,
                 recipe="depth", stage=None, sky=False, sky_chunk=None, sky_limit=None, warn=None, sky_scaled=True):
    """stage(name, done=None, total=None): the step names, each sent BEFORE its call; None =
    silent. Progress only: no value here depends on it.
    sky=True: the sky repair after the merge (default off: only the app's CLI turns it on). sky_chunk = frames per model
    run, sky_limit = most frames for one photo, warn(msg) = one line for the user when the repair fails. A failed repair
    never costs the map: the photo keeps the unrepaired one."""
    staged = stage is not None
    stage = stage or (lambda *a, **k: None)
    H, W = photo.shape[:2]
    info = dict(canvas=[W, H], tile=tile)
    R = TC.RECIPES[recipe]

    mw, mh, mp = TC.pin_for16(W, H, m1_budget, R["m1_strip"], R["m1_max_long_edge"])
    if mw is None:
        raise ValueError(f"{W}x{H}: no global grid fits the {m1_budget} MP budget")
    m1_decoy = TC.load_decoy(R["m1_decoy_scale"]) if R["m1_decoy_scale"] > 0 else None
    m1_native = TC.build_m1_input(photo, mw, mh, m1_decoy, R["m1_strip"])
    iw, ih = m1_native.shape[1], m1_native.shape[0]
    assert iw % 16 == 0 and ih % 16 == 0, (iw, ih)
    info.update(m1_grid=[mw, mh], m1_input=[iw, ih], m1_mp=round(mp, 2))
    log(f"[tiled] {recipe} model | photo {W}x{H} | global {mw}x{mh} + {R['m1_strip']}-px strip "
        f"({'hand' if m1_decoy else 'no hand'}) = {iw}x{ih} ({mp:.2f} MP, /16 grid)")
    if m1_decoy is not None:
        hand_h = m1_decoy["obj"].shape[0]
        if hand_h + TC.DECOY_BOTTOM > mh:
            log(f"[tiled] wide photo: the global pass's hand is scaled down to fit ({hand_h} -> {mh - 2 * TC.DECOY_BOTTOM} px tall)")

    cover = (mw * mh / (W * H)) ** 0.5
    info["global_cover"] = round(cover, 3)
    covered = cover >= GLOBAL_COVER_MIN
    single = W < tile or H < tile or covered
    if single:
        plan, inputs = None, [m1_native]
        info["mode"] = (f"single pass (photo smaller than one {tile}px tile)" if (W < tile or H < tile) else
                        f"single pass (the global pass sees the photo at {cover:.3f} of its size >= {GLOBAL_COVER_MIN})")
        log(f"[tiled] {info['mode']}: the global pass alone, no tiles")
    else:
        plan = TC.make_plan(W, H, tile, TC.OVERLAP)
        stage(f"preparing {len(plan['tiles']) + 1} tiles")
        tins = TC.build_tile_inputs(photo, plan, TC.load_decoy(TC.TILE_DECOY_SCALE))
        inputs = [m1_native] + tins
        del tins
        info["mode"] = f"tiled {plan['cols']}x{plan['rows']} = {len(plan['tiles'])} tiles"
        log(f"[tiled] plan: tile {tile} (+ {TC.TILE_STRIP}-px decoy strip) overlap min {TC.OVERLAP} -> {info['mode']} | "
            f"achieved overlap {plan['achieved_overlap_x']:.0f} / {plan['achieved_overlap_y']:.0f} px")

    t0 = time.time()
    preds = infer(inputs, progress, stage) if staged else infer(inputs, progress)    # a0_engine's stand-in takes two
    info["infer_s"] = round(time.time() - t0, 1)
    assert len(preds) == len(inputs), (len(preds), len(inputs))
    del inputs

    # ---- the global: strip cropped, resized to the canvas
    p1 = preds[0]
    assert p1.shape == (ih, iw), (p1.shape, (ih, iw))
    m1 = TC.crop_strip(p1, mw)
    stage("placing the overview")
    G = TC.global_from_m1(m1, W, H)
    info["m1_near_rail_pct"] = round(100 * float((m1 <= -TC.SAT_THR).mean()), 3)
    log(f"[tiled] global prediction {mw}x{mh} range {m1.min():.4f}..{m1.max():.4f} | near rail {info['m1_near_rail_pct']} % "
        f"-> resized to the photo {W}x{H}, range {G.min():.4f}..{G.max():.4f}")

    if single:
        S = G
    else:
        tp = [TC.crop_strip(p, tile) for p in preds[1:]]
        del preds
        nt, fitted = len(tp), [0]

        def log_fit(msg):                   # the engine's one line per fitted tile -> "fitting tiles k/n" (no engine edit)
            if str(msg).startswith("[tiled_core] tile_"):
                fitted[0] += 1
                stage(f"fitting tiles {fitted[0]}/{nt}", fitted[0], nt)
            log(msg)
        stage(f"fitting tiles 0/{nt}", 0, nt)
        Tm, rows = TC.mosaic(G, plan, tp, log=log_fit)
        stage("merging the tiles")
        S, minfo = TC.laplacian_merge(G, Tm, TC.RADIUS, merge_device)
        del Tm
        info["tiles_near_rail_max_pct"] = round(100 * max(r["near"] for r in rows), 3)
        info["affine_fallback"] = [f"{r['name']} ({r['steep_x_median']:.0f}x median)" for r in rows if r["affine_fallback"]]
        if info["affine_fallback"]:
            log(f"[tiled] steep-segment fallback (straight-line fit) on: {', '.join(info['affine_fallback'])}")
        log(f"[tiled] merge r{TC.RADIUS} on {merge_device}: {minfo['levels']} levels, residual {minfo['residual_px']} px | "
            f"range {S.min():.4f}..{S.max():.4f}")
        if sky:
            S = _sky(S, G, plan, tp, rows, photo, infer, merge_device, stage, log, sky_chunk, sky_limit, warn, info, sky_scaled)
        del tp
    stage("finishing the map")
    out, sinfo = TC.stretch_near_bright(S)
    info["stretch"] = dict(lo=round(sinfo["lo"], 5), hi=round(sinfo["hi"], 5))
    log(f"[tiled] stretch 0.5 / 99.5 %: {sinfo['lo']:.4f}..{sinfo['hi']:.4f} -> 0..1, near = bright | "
        f"clipped {100 * sinfo['clipped_low']:.2f} % / {100 * sinfo['clipped_high']:.2f} %")
    return out, info
