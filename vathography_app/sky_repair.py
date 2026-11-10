# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""sky_repair.py — repair of tiles the model misread (typically a web of cables over a sky read as a ceiling), run
inside the pipeline after the merge, on arrays in memory.

  1. FLAG     tiles whose prediction does not follow the global map: correlation to G <= CORR_MAX, or fit error
              >= RMS_X x the photo's median, or gain to G <= GAIN_MIN. Nothing flagged: repair() returns None and the
              map stays exactly as it was.
  2. FRAMES   K candidate crops per flagged tile, no decoy strip, +-SEARCH px, ranked by G's structure energy x
              (1 - 0.5 sky fraction), at least MIN_SEP px apart, then one extra model run.
  3. PICK     on the tile's footprint inside the frame: gain to G > GAIN_MIN and photo-edge recall >= RECALL_MIN; best
              recall wins.
  4. SCALED   one more input per flagged tile, past the frame limit too: a SCALED_SIDE square around it brought down to
              one tile, enlarged back, cropped to the tile, same scores. Left out on photos shorter than SCALED_SIDE, where
              the scale could not hold. Neither a frame nor a scaled frame qualifies: the tile keeps its grid prediction.
  5. MERGE    local fit, photo sky mask, frames at weight 1 (feathered, not on canvas sides), scaled frames below them,
              the replaced grid tiles keep what those two leave, missing weight = G clamped to the map's range, Laplacian
              merge.
  6. LINES    the photo-guided G on the photo's thin dark lines near large skies, where it is nearer.
  7. SCOPE    the result replaces the map only inside the flagged tiles + their grid neighbours + the chosen frames
              (raised-cosine ramp inside that union); outside it the map, before the final stretch, is kept as it was.
"""
import math
import time
import warnings

import cv2
import numpy as np
import torch

from .engine import tiled_core as TC

CORR_MAX, RMS_X, GAIN_MIN = 0.8, 3.8, 0.05           # the flag rule
MIN_VALID = 64                                       # fewer valid pixels than this: no fit on that tile or frame
K, SEARCH, MARGIN, COVER, MIN_SEP = 6, 768, 192, 0.4, 128
RECALL_MIN, EDGE_Q, EDGE_T = 0.1, 0.05, 0.004
R_WIN, SIG_LP, EPS, TAU = 64, 3.0, 1e-4, 0.15
FLAT, FLAT_W = 0.8, 61
FR_PRIO, FR_FEATHER = 1.0, 256
MERGE_R = 128
BH, BH_T = 21, 0.06                                  # thin dark lines: black-hat size + threshold
SKY_CC_FRAC = 50000 / (6656 * 9984)                  # a "large" sky component, as a share of the canvas
SCOPE_RAMP = 256
SCALED_SIDE = 2775                                   # scaled frame: a square of this side, brought down to one tile
LIMIT_SHARE = 0.4                                    # full frames on at most this share of a photo's tiles, worst first


def _grad(a, s):
    b = cv2.GaussianBlur(a, (0, 0), s)
    return cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)) / 8.0


def _fr_weight(n, f):
    r = np.ones(n, np.float32); t_ = np.linspace(0, 1, f, dtype=np.float32); ramp = 0.5 - 0.5 * np.cos(np.pi * t_)
    r[:f] = ramp; r[n - f:] = np.minimum(r[n - f:], ramp[::-1]); return r


def _frame_w(x0, y0, tile, W, H):
    fr = _fr_weight(tile, FR_FEATHER); rx, ry = fr.copy(), fr.copy()
    if x0 == 0: rx[:FR_FEATHER] = 1.0
    if x0 + tile == W: rx[tile - FR_FEATHER:] = 1.0
    if y0 == 0: ry[:FR_FEATHER] = 1.0
    if y0 + tile == H: ry[tile - FR_FEATHER:] = 1.0
    return ry[:, None] * rx[None, :]


def _fit_mask(p, tile):
    cx0 = int(tile * (1 - TC.FIT_CENTRAL) / 2); cx1 = tile - cx0
    kern = np.ones((2 * TC.SAT_DILATE_PX + 1, 2 * TC.SAT_DILATE_PX + 1), np.uint8)
    v = ~cv2.dilate((np.abs(p) >= TC.SAT_THR).astype(np.uint8), kern).astype(bool)
    v[:cx0, :] = False; v[cx1:, :] = False; v[:, :cx0] = False; v[:, cx1:] = False
    return v


def _gain(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # a nearly flat tile: numpy warns, the value stands
        return float(np.polyfit(a, b, 1)[0])


def flag_tiles(G, plan, tp, rows, log=print):
    """Returns [(tile dict, corr)], one report line per flagged or skipped tile."""
    tile = plan["tile"]
    med = float(np.median([r["rms"] for r in rows]))
    out = []
    for t, p, r in zip(plan["tiles"], tp, rows):
        v = _fit_mask(p, tile)
        a, b = p[v][::7], G[t["y0"]:t["y1"], t["x0"]:t["x1"]][v][::7]
        if len(a) < MIN_VALID or float(np.ptp(a)) < 1e-6:
            log(f"[sky_repair] {t['name']}: too few valid pixels, skipped")
            continue
        gain, rx = _gain(a, b), r["rms"] / med
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = float(np.corrcoef(a, b)[0, 1])
        if corr <= CORR_MAX or rx >= RMS_X or gain <= GAIN_MIN:       # a NaN corr compares False: not flagged by it
            out.append((t, corr))
            log(f"[sky_repair] FLAG {t['name']}: corr {corr:.3f} | rms {rx:.1f}x median | gain {gain:+.3f}")
    return out


def _candidates(G, lum, plan, flagged, log):
    H, W = G.shape; tile = plan["tile"]
    E = cv2.magnitude(cv2.Sobel(cv2.GaussianBlur(G, (0, 0), 2), cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(cv2.GaussianBlur(G, (0, 0), 2), cv2.CV_32F, 0, 1, ksize=3))
    mu = cv2.GaussianBlur(lum, (0, 0), 3); sd = np.sqrt(np.maximum(cv2.GaussianBlur(lum * lum, (0, 0), 3) - mu * mu, 0))
    Ei = cv2.integral(E.astype(np.float64)); Si = cv2.integral(((lum > 0.55) & (sd < 0.02)).astype(np.float64))
    box_i = lambda Ii, x0, y0, x1, y1: Ii[y1, x1] - Ii[y0, x1] - Ii[y1, x0] + Ii[y0, x0]

    def inner_fp(fx, fy, t):
        ix0, iy0 = max(fx + MARGIN, t["x0"]), max(fy + MARGIN, t["y0"])
        ix1, iy1 = min(fx + tile - MARGIN, t["x1"]), min(fy + tile - MARGIN, t["y1"])
        return None if ix1 <= ix0 or iy1 <= iy0 else (ix0, iy0, ix1, iy1)
    cands = []
    for t in flagged:
        fpa = (t["x1"] - t["x0"]) * (t["y1"] - t["y0"]); pool = []
        for fy in range(max(0, t["y0"] - SEARCH), min(H - tile, t["y0"] + SEARCH) + 1, 64):
            for fx in range(max(0, t["x0"] - SEARCH), min(W - tile, t["x0"] + SEARCH) + 1, 64):
                fp = inner_fp(fx, fy, t)
                if fp is None or (fp[2] - fp[0]) * (fp[3] - fp[1]) < COVER * fpa:
                    continue
                skyf = box_i(Si, fx, fy, fx + tile, fy + tile) / tile / tile
                pool.append((box_i(Ei, *fp) * (1.0 - 0.5 * skyf), fx, fy))
        pool.sort(key=lambda r: -r[0]); pick = []
        for _, fx, fy in pool:
            if len(pick) >= K:
                break
            if all(max(abs(fx - q["x0"]), abs(fy - q["y0"])) >= MIN_SEP for q in pick):
                pick.append(dict(src=t["name"], x0=fx, y0=fy, fp=inner_fp(fx, fy, t)))
        log(f"[sky_repair] candidates {t['name']}: pool {len(pool)} -> {[(q['x0'], q['y0']) for q in pick]}")
        cands += pick
    return cands


def _run_frames(cands, bgr, G, lum, plan, infer, chunk, stage, log, total):
    """One model run in chunks of one weight pass; each frame is scored at once and only the best per tile is kept, so
    memory does not grow with the frame count. Progress counts toward `total`, the frames + the scaled frames. Returns
    {tile name: dict(pred, x0, y0, why)}."""
    tile = plan["tile"]
    EL = _grad(lum, 2)
    best = {}
    chunk = max(1, int(chunk or len(cands)))
    n = len(cands); parts = -(-n // chunk)
    stage("repairing the sky", 0, total)
    t0 = time.time()
    for g in range(0, n, chunk):
        part = cands[g:g + chunk]

        def tick(d, tt, g=g, m=len(part)):                 # this stage's own count, rising over all the chunks
            stage("repairing the sky", g + int(m * d / max(tt, 1)), total)
        preds = infer([bgr[q["y0"]:q["y0"] + tile, q["x0"]:q["x0"] + tile].copy() for q in part], tick)
        for q, p in zip(part, preds):
            p = np.ascontiguousarray(np.asarray(p).squeeze().astype(np.float32)[:, :tile])
            ix0, iy0, ix1, iy1 = q["fp"]; fx, fy = q["x0"], q["y0"]
            sub = (slice(iy0 - fy, iy1 - fy), slice(ix0 - fx, ix1 - fx))
            v = _fit_mask(p, tile); g_ = G[fy:fy + tile, fx:fx + tile]
            a, b = p[sub][v[sub]][::5], g_[sub][v[sub]][::5]
            if len(a) < MIN_VALID or float(np.ptp(a)) < 1e-6:
                log(f"[sky_repair]    frame ({fx},{fy}) for {q['src']}: too few valid pixels, no")
                continue
            gain = _gain(a, b)
            lo, hi = np.percentile(p, [1, 99]); ep = _grad(p, 2)[sub] / max(hi - lo, 1e-6)
            el = EL[iy0:iy1, ix0:ix1]; strong = el >= np.quantile(el, 1 - EDGE_Q)
            rec = float((ep[strong] >= EDGE_T).mean())
            ok = gain > GAIN_MIN and rec >= RECALL_MIN
            log(f"[sky_repair]    frame ({fx},{fy}) for {q['src']}: gain {gain:+.3f} | recall {rec:.3f} | {'qualifies' if ok else 'no'}")
            if ok and (q["src"] not in best or rec > best[q["src"]]["rec"]):
                best[q["src"]] = dict(pred=p, x0=fx, y0=fy, rec=rec, why=f"recall {rec:.3f}, gain {gain:+.3f}")
        del preds
    log(f"[sky_repair] frames: {n} in {parts} part(s), {time.time() - t0:.0f} s")
    return best


def _run_scaled(flagged, bgr, G, lum, plan, infer, chunk, stage, log, base, total):
    """One extra model run per flagged tile: a wider square around the tile brought down to one tile, enlarged back and
    cropped to the tile. Same scores and thresholds as a frame. Progress goes on from `base`. Returns {tile name:
    prediction at the tile's position}."""
    H, W = G.shape; tile = plan["tile"]
    side = SCALED_SIDE
    EL = _grad(lum, 2)
    out = {}
    chunk = max(1, int(chunk or len(flagged)))
    n = len(flagged); parts = -(-n // chunk)
    t0 = time.time()
    for g in range(0, n, chunk):
        part = flagged[g:g + chunk]
        spots = [(max(0, min(W - side, t["x0"] + tile // 2 - side // 2)), max(0, min(H - side, t["y0"] + tile // 2 - side // 2))) for t in part]

        def tick(d, tt, g=g, m=len(part)):
            stage("repairing the sky", base + g + int(m * d / max(tt, 1)), total)
        preds = infer([cv2.resize(bgr[ry:ry + side, rx:rx + side], (tile, tile), interpolation=cv2.INTER_AREA) for rx, ry in spots], tick)
        for t, (rx, ry), p in zip(part, spots, preds):
            p = cv2.resize(np.ascontiguousarray(np.asarray(p).squeeze().astype(np.float32)[:, :tile]), (side, side), interpolation=cv2.INTER_CUBIC)
            fx, fy = t["x0"], t["y0"]
            p = p[fy - ry:fy - ry + tile, fx - rx:fx - rx + tile].copy()
            sub = (slice(MARGIN, tile - MARGIN), slice(MARGIN, tile - MARGIN))
            v = _fit_mask(p, tile); g_ = G[fy:fy + tile, fx:fx + tile]
            a, b = p[sub][v[sub]][::5], g_[sub][v[sub]][::5]
            if len(a) < MIN_VALID or float(np.ptp(a)) < 1e-6:
                log(f"[sky_repair]    scaled frame for {t['name']}: too few valid pixels, no")
                continue
            gain = _gain(a, b)
            lo, hi = np.percentile(p, [1, 99]); ep = _grad(p, 2)[sub] / max(hi - lo, 1e-6)
            el = EL[fy + MARGIN:fy + tile - MARGIN, fx + MARGIN:fx + tile - MARGIN]; strong = el >= np.quantile(el, 1 - EDGE_Q)
            rec = float((ep[strong] >= EDGE_T).mean())
            ok = gain > GAIN_MIN and rec >= RECALL_MIN
            log(f"[sky_repair]    scaled frame for {t['name']}: gain {gain:+.3f} | recall {rec:.3f} | {'qualifies' if ok else 'no'}")
            if ok:
                out[t["name"]] = p
        del preds
    log(f"[sky_repair] scaled frames: {n} in {parts} part(s), {len(out)} qualify, {time.time() - t0:.0f} s")
    return out


def repair(S, G, plan, tp, rows, bgr, infer, merge_device="cpu", stage=None, log=print, chunk=None, limit=None, scaled=True, _flags=None):
    """Returns None (nothing flagged) or (final float32 map, report dict). `chunk` = frames per model run (one weight
    pass); `limit` = at most this many full frames, worst tiles first, one tile's frames at the least; the others are
    named in the log. `scaled` = the low-priority scaled frame on every flagged tile: a
    full-resolution frame wins where it covers, the scaled frame fills what it leaves, the grid tile only what is left
    after that. `_flags` = tile names to use in place of the rule (a gate's forced flags)."""
    stage = stage or (lambda *a, **k: None)
    t_all = time.time()
    H, W = S.shape; tile = plan["tile"]; T = plan["tiles"]
    assert bgr.shape[:2] == (H, W), (bgr.shape, (H, W))
    if _flags is None:
        scored = flag_tiles(G, plan, tp, rows, log)
    else:
        scored = [(t, 0.0) for t in T if t["name"] in set(_flags)]
    if not scored:
        log("[sky_repair] no tile flagged -> the map is kept unchanged")
        return None
    past = []
    every = [s[0] for s in scored]                   # the scaled frames go to every flagged tile, past the limit too
    if limit is None:
        limit = K * math.ceil(LIMIT_SHARE * len(T))
    if K * len(scored) > limit:
        keep = max(int(limit) // K, 1)
        worst = sorted(scored, key=lambda s: (s[1] != s[1], s[1]))       # lowest correlation first, NaN last
        names = {s[0]["name"] for s in worst[:keep]}
        past = [s[0]["name"] for s in scored if s[0]["name"] not in names]
        scored = [s for s in scored if s[0]["name"] in names]
        log(f"[sky_repair] frame limit {limit}: full frames on {keep} tile(s), past the limit: {past}")
    flagged = [s[0] for s in scored]
    lum = bgr.astype(np.float32).mean(2) / 255
    log(f"[sky_repair] {len(every)} tile(s) flagged, canvas {W}x{H}, tile {tile}, map range {S.min():.4f}..{S.max():.4f}")

    torch.cuda.empty_cache()                         # the first pass ran in this process
    cands = _candidates(G, lum, plan, flagged, log)
    scaled = scaled and min(H, W) >= SCALED_SIDE
    if not scaled:
        log(f"[sky_repair] no scaled frames: off, or the photo's short side is under {SCALED_SIDE} px")
    total = len(cands) + (len(every) if scaled else 0)
    chosen = _run_frames(cands, bgr, G, lum, plan, infer, chunk, stage, log, total) if cands else {}
    soft = _run_scaled(every, bgr, G, lum, plan, infer, chunk, stage, log, len(cands), total) if scaled else {}
    torch.cuda.empty_cache()
    for t in every:
        c = chosen.get(t["name"])
        pick = f"frame ({c['x0']},{c['y0']}), {c['why']}" if c else "scaled frame only" if t["name"] in soft else "none qualifies, its grid tile stays"
        log(f"[sky_repair] PICK {t['name']}: {pick}")

    # ---- the merge
    stage("merging the repair")
    drop = set(chosen) | set(soft)
    mu = cv2.GaussianBlur(lum, (0, 0), 3); sd = np.sqrt(np.maximum(cv2.GaussianBlur(lum * lum, (0, 0), 3) - mu * mu, 0))
    SKYM = ((lum > 0.55) & (sd < 0.02)).astype(np.uint8)
    SKYM = cv2.morphologyEx(SKYM, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    SKY_OPEN = cv2.erode(SKYM, np.ones((25, 25), np.uint8)).astype(np.float32)
    SKY_LUM = cv2.blur(lum * SKY_OPEN, (129, 129)) / np.maximum(cv2.blur(SKY_OPEN, (129, 129)), 1e-6)
    box = lambda x: cv2.blur(x, (2 * R_WIN + 1, 2 * R_WIN + 1), borderType=cv2.BORDER_REFLECT)

    def local_fit(p, g):
        pl = cv2.GaussianBlur(p, (0, 0), SIG_LP); mp, mg = box(pl), box(g)
        a = (box(pl * g) - mp * mg) / (box(pl * pl) - mp * mp + EPS); b = mg - a * mp
        return box(a) * p + box(b)
    records = [dict(t, kind="grid", pred=p) for t, p in zip(T, tp)]
    for src in [t["name"] for t in flagged if t["name"] in chosen]:
        c = chosen[src]
        records.append(dict(name=f"framed@{c['x0']},{c['y0']}", kind="framed", pred=c["pred"], x0=c["x0"], y0=c["y0"],
                            x1=c["x0"] + tile, y1=c["y0"] + tile, i=-1, j=-1))
    for t in every:
        if t["name"] in soft:
            records.append(dict(name=f"scaled@{t['x0']},{t['y0']}", kind="scaled", pred=soft[t["name"]], x0=t["x0"], y0=t["y0"],
                                x1=t["x1"], y1=t["y1"], i=-1, j=-1))
    FCOV = np.zeros((H, W), np.float32); SCOV = np.zeros((H, W), np.float32)
    for t in records:
        if t["kind"] != "grid":
            (FCOV if t["kind"] == "framed" else SCOV)[t["y0"]:t["y1"], t["x0"]:t["x1"]] += _frame_w(t["x0"], t["y0"], tile, W, H)
    FCOV = np.clip(FCOV, 0, 1)
    COV = FCOV + (1 - FCOV) * np.clip(SCOV, 0, 1)          # what the grid tiles of the replaced tiles give up
    del SCOV
    num = np.zeros((H, W), np.float32); den = np.zeros((H, W), np.float32)
    t0 = time.time()
    for t in records:
        p = np.ascontiguousarray(t["pred"].astype(np.float32)[:, :tile])
        g = G[t["y0"]:t["y1"], t["x0"]:t["x1"]]
        sl0 = (slice(t["y0"], t["y1"]), slice(t["x0"], t["x1"]))
        valid = _fit_mask(p, tile)
        if int(valid.sum()) < MIN_VALID or float(np.ptp(p[valid])) < 1e-6:
            log(f"[sky_repair] {t['name']}: too few valid pixels, left out of the merge")
            continue
        kt, kg, *_ = TC.quantile_map(p[valid], g[valid], TC.KNOT_Q)
        q = TC.apply_quantile_map(p, kt, kg)
        L = local_fit(p, g).astype(np.float32)
        mo = SKY_OPEN[sl0]
        cnt = box(mo); ls = box(mo * g) / np.maximum(cnt, 1e-6)
        ns = (1 - SKYM[sl0]).astype(np.float32)
        lum0 = lum[sl0]; core0 = cv2.erode(ns, np.ones((7, 7), np.uint8))
        kb = lambda f: cv2.blur(f, (FLAT_W, FLAT_W), borderType=cv2.BORDER_REFLECT)
        cl = np.where(kb(core0) > 1e-3, kb(core0 * lum0) / np.maximum(kb(core0), 1e-6), 0.0)
        edge_sky = (ns > 0) & (lum0 > 0.5 * (cl + SKY_LUM[sl0])) & (cnt > 0.02)
        ns = np.where(edge_sky, 0.0, ns).astype(np.float32)
        sw = ((SKYM[sl0] > 0) | edge_sky).astype(np.float32) * (cnt > 0.02)
        core = cv2.erode(ns, np.ones((7, 7), np.uint8))
        cc = kb(core); lvl = np.where(cc > 1e-3, kb(core * L) / np.maximum(cc, 1e-6), L)
        L = np.where(ns > 0, lvl + FLAT * (L - lvl), L).astype(np.float32)
        L = (sw * ls + (1 - sw) * L).astype(np.float32)
        wd = np.exp(-(cv2.GaussianBlur(q - g, (0, 0), 32.0) / TAU) ** 2).astype(np.float32)
        L2 = wd * q + (1 - wd) * L
        if t["kind"] != "grid":
            L2 = (sw * L + (1 - sw) * L2).astype(np.float32)
            w = (FR_PRIO * _frame_w(t["x0"], t["y0"], tile, W, H)).astype(np.float32)
            if t["kind"] == "scaled":
                w = w * (1.0 - FCOV[sl0])
        else:
            wx = TC.ramp_weights(tile, t["i"] > 0, t["i"] < plan["cols"] - 1, plan["achieved_overlap_x"], plan["achieved_overlap_x"])
            wy = TC.ramp_weights(tile, t["j"] > 0, t["j"] < plan["rows"] - 1, plan["achieved_overlap_y"], plan["achieved_overlap_y"])
            w = wy[:, None] * wx[None, :]
            if t["name"] in drop:
                w = w * (1.0 - COV[sl0])
        num[sl0] += w * L2; den[sl0] += w
    log(f"[sky_repair] merge: {len(records)} tiles fitted in {time.time() - t0:.0f} s | replaced grid tiles {sorted(drop) or 'none'} | "
        f"missing weight on {100 * (den < 0.999).mean():.2f} %")
    Gf = np.clip(G, float(S.min()), float(S.max()))
    Tm = ((num + np.clip(1 - den, 0, 1) * Gf) / np.maximum(den, 1.0)).astype(np.float32)
    del num, den, records, FCOV, COV
    torch.cuda.empty_cache()
    F = TC.laplacian_merge(G, Tm, MERGE_R, merge_device)[0].astype(np.float32)
    del Tm
    torch.cuda.empty_cache()
    log(f"[sky_repair] laplacian merge r{MERGE_R}: range {F.min():.4f}..{F.max():.4f}")

    # ---- thin dark lines near a large sky
    stage("sky lines")
    ccmin = int(round(SKY_CC_FRAC * W * H))
    skyfull = ((lum > 0.55) & (sd < 0.02)).astype(np.uint8)
    sk = cv2.morphologyEx(skyfull, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(sk, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), 8)
    big = [i for i in range(1, n_) if st_[i, 4] >= ccmin]
    y1 = min(H, int(max(st_[i, 1] + st_[i, 3] for i in big)) + 61) if big else 0
    lo_c, hi_c = float(S.min()), float(S.max())
    out = F.copy()
    if y1 > 0:
        from .sky_fbs import fbs_tiled
        t0 = time.time()
        lumb = bgr[:y1].astype(np.float32).mean(2) / 255
        bh = cv2.morphologyEx(lumb, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BH, BH)))
        mub = cv2.GaussianBlur(lumb, (0, 0), 3); sdb = np.sqrt(np.maximum(cv2.GaussianBlur(lumb * lumb, (0, 0), 3) - mub * mub, 0))
        sky = ((lumb > 0.55) & (sdb < 0.02)).astype(np.uint8)
        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), 8)
        keep_ = np.zeros(n_, bool); keep_[1:] = st_[1:, 4] >= ccmin
        sky = (keep_[lab_] & (sky > 0)).astype(np.uint8)
        near_sky = cv2.dilate(sky, np.ones((61, 61), np.uint8)).astype(bool)
        line = cv2.dilate(((bh > BH_T) & near_sky).astype(np.uint8), np.ones((3, 3), np.uint8))
        rgb = np.ascontiguousarray(bgr[:y1, :, ::-1]); g = G[:y1].astype(np.float32)
        gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        conf = np.clip(1.0 - gm / (np.percentile(gm, 99) + 1e-6), 0.05, 1.0).astype(np.float32)
        Gg, solved = fbs_tiled(rgb, g, conf, line > 0)
        del rgb, conf, gm
        M = cv2.GaussianBlur(line.astype(np.float32), (0, 0), 1.0)
        M = M * (Gg < F[:y1] - 0.02)
        out[:y1] = (M * Gg + (1 - M) * F[:y1]).astype(np.float32)
        out[:y1] = np.where(M > 0, np.clip(out[:y1], lo_c, hi_c), out[:y1]).astype(np.float32)
        log(f"[sky_repair] sky lines: band of {y1} px | {solved} tile(s) solved | {time.time() - t0:.0f} s | line mask {100 * float((M > 0.5).mean()):.3f} %")
    else:
        log("[sky_repair] sky lines: no large sky, skipped")

    # ---- scope: flagged tiles + grid neighbours + frames
    U = np.zeros((H, W), np.uint8)
    fl_ij = {(t["i"], t["j"]) for t in every}
    for t in T:
        if any(abs(t["i"] - i) <= 1 and abs(t["j"] - j) <= 1 for i, j in fl_ij):
            U[t["y0"]:t["y1"], t["x0"]:t["x1"]] = 1
    for c in chosen.values():
        U[c["y0"]:c["y0"] + tile, c["x0"]:c["x0"] + tile] = 1
    # distance to the nearest pixel outside the union; padding with 1: a union side ON the canvas border gets no ramp
    U_b = cv2.distanceTransform(np.pad(U, 1, constant_values=1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    x = np.clip(U_b / SCOPE_RAMP, 0, 1); Ms = (0.5 - 0.5 * np.cos(np.pi * x)).astype(np.float32) * (U > 0)
    final = (S * (1 - Ms) + out * Ms).astype(np.float32)
    d = np.abs(final - S)
    log(f"[sky_repair] scope {100 * float((U > 0).mean()):.1f} % of the canvas | changed > 0.01 on {100 * float((d > 0.01).mean()):.2f} % | "
        f"total {time.time() - t_all:.0f} s")
    return final, dict(flagged=[t["name"] for t in every], past_limit=past, scaled=sorted(soft),
                       chosen={k: (v["x0"], v["y0"]) for k, v in chosen.items()}, seconds=round(time.time() - t_all, 1))
