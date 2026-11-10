# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""relief_app.py — the Relief tab's core = node 1.3.0's "Depth Relief Increase - Vathography" without ComfyUI.

What the node asks the user, the app reads from the file: the app's own depth files carry the
`vathography = raw-depth` tag -> "Raw depth map"; untagged -> an add-on export by extension (.exr -> "Blender export,
EXR (linear)", .png -> "Blender export, PNG (sRGB)"). The picture is read exactly as LoadImage reads it
(photo_io.read_image_float), then relief_node.execute's logic per picture (grey test GREY_TOL, NaN check,
RL.relief / colour path) — copied from the pinned 1.3.0 relief_node.py:53-135, ComfyUI calls removed.

Device policy: the node puts everything on its GPU. Under the app's budget:
  grey    48 B/px (node constant) fits -> GPU (66.5 MP measured 3.6 GiB) ; else CPU
  colour  48 + 160 B/px fits -> all on the GPU, as the node (identical) ;
          else SPLIT: the float64 colour maths on the CPU, the Local Laplacian (RL.reshape) on the GPU — the node's
          relief_colour code with ONE change, the device of the reshape call (all-CPU took 21 min at 66.5 MP, the
          reshape alone 20 min)
Output = the same format as the input, `<stem>_relief_strength<N>_vNNN.<ext>` (never over an existing file; older versions
wrote `<stem>_relief_strength<N>.<ext>` / `_relief_level<N>`, counted as version 0),
tagged only when the input was.
"""
import os

import numpy as np
import torch

from . import outnames, photo_io
from .engine import relief as RL
from .engine import relief_colour as RC

GREY_TOL = 1e-6                # = relief_node.GREY_TOL
WORK_BYTES_PER_PX = 48         # = relief_node.WORK_BYTES_PER_PX
COLOUR_BYTES_PER_PX = 160      # = relief_node.COLOUR_BYTES_PER_PX


def source_for(path):
    if photo_io.read_tag(path) == photo_io.TAG_VALUE:
        return RL.SOURCES[2]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        return RL.SOURCES[0]
    if ext == ".png":
        return RL.SOURCES[1]
    raise ValueError(f"{os.path.basename(path)}: the Relief tab takes the add-on's PNG or EXR export, or a depth map "
                     "made by this app")


def relief_colour_split(rgb, sigma_r, source, colour_device, llf_device, tick=None, stage=None):
    """= engine relief_colour (1.3.0, relief_colour.py:178-204) with the reshape on `llf_device`; identical when the
    two devices are the same."""
    if source == RL.SOURCES[2]:
        raise ValueError("'Raw depth map' takes a grey depth map; for a colour export set 'input' to the export's "
                         "format (PNG or EXR)")
    H, W, _ = rgb.shape
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=colour_device, dtype=torch.float64).reshape(-1, 3)
    lin = x if source == RL.SOURCES[0] else torch.from_numpy(RL.eotf_srgb_ext(rgb).reshape(-1, 3)).to(x)
    if stage:
        stage("fitting the colour ramp")
    with torch.no_grad():
        lab = RC.to_oklab(lin)
        poly = RC.fit_curve(lab.float()).double()
        t, dist = RC._project(lab, poly)
    off_share = float((dist > RC.OFF_DIST).double().mean())
    info = dict(path="curve", off_share=off_share, curve_len=float((poly[1:] - poly[:-1]).norm(dim=1).sum()))
    if off_share > RC.OFF_SHARE_MAX:
        info["path"] = "brightness"                                     # = RC._relief_brightness, reshape on llf_device
        Y = (lin @ RC._LUMA.to(lin)).clamp(min=0)
        disp = RL.oetf_srgb_ext(Y.reshape(H, W).cpu().numpy())
        r, rinfo = RL.reshape(disp, sigma_r, device=llf_device, tick=tick)
        r, rinfo["levels_kept"] = RL.keep_levels(r, disp)
        info.update(rinfo)
        Yn = torch.from_numpy(RL.eotf_srgb_ext(r).reshape(-1)).to(lin)
        gain = Yn / Y.clamp(min=1e-4)
        return RC._encode(lin * gain[:, None], H, W, source, info)
    t_np = t.reshape(H, W).float().cpu().numpy()
    r, rinfo = RL.reshape(t_np, sigma_r, device=llf_device, tick=tick)
    r, rinfo["levels_kept"] = RL.keep_levels(r, t_np)
    info.update(rinfo)
    with torch.no_grad():
        tn = torch.from_numpy(r.reshape(-1)).to(lab)
        out_lin = RC.from_oklab(lab + RC._at(poly, tn) - RC._at(poly, t))
    return RC._encode(out_lin, H, W, source, info)


def file_is_grey(path):
    """Grey vs colour decided from the FILE's own channels: core LoadImage's PyAV decode of a
    16-bit PNG / float EXR adds per-channel differences up to 9e-5 / 1.8e-4 to a file whose channels are identical,
    which the node's 1e-6 test reads as colour. PNG and EXR are read exactly with OpenCV here;
    any other file falls back to the decoded picture (8-bit decodes keep equal channels)."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.ndim == 2:
        return True
    a = a[..., :3]
    return bool(np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]))


def devices_for(H, W, colour, budget_gib, gpu):
    """-> (colour_device, llf_device) under the budget; gpu = None means no CUDA."""
    if gpu is None:
        return "cpu", "cpu"
    budget = budget_gib * 2 ** 30
    if not colour:
        return None, (gpu if H * W * WORK_BYTES_PER_PX <= budget else "cpu")
    if H * W * (WORK_BYTES_PER_PX + COLOUR_BYTES_PER_PX) <= budget:
        return gpu, gpu
    return "cpu", (gpu if H * W * WORK_BYTES_PER_PX <= budget else "cpu")


def relief_image(img, level, source, colour_device, llf_device, log=print, grey=None, stage=None):
    """img float32 [H, W, 3] as LoadImage gives it -> (out float32 [H, W] grey or [H, W, 3] colour, info).
    = relief_node.execute's per-image body (1.3.0 relief_node.py:62-98, :105-135). grey = the file's own verdict
    (file_is_grey); None = the node's test on the decoded picture."""
    sigma_r = RL.LEVELS[int(level)]
    # the engine's own sample counter (engine relief.py fast_llf_beta: tick(j + 1, ns)) -> the row's progress
    tick = (lambda j, n: stage(f"accentuating · sample {j} of {n}", j, n)) if stage else None
    spread = float(max(np.abs(img[..., 0] - img[..., 1]).max(), np.abs(img[..., 0] - img[..., 2]).max()))
    if (not grey) if grey is not None else spread > GREY_TOL:
        log(f"[relief] colour picture (channels differ by up to {spread:.4f}) -> relief along its colour ramp "
            f"(colour maths on {colour_device}, filter on {llf_device})")
        r, info = relief_colour_split(img[..., :3], sigma_r, source, colour_device, llf_device, tick=tick, stage=stage)
        return r, info
    d = np.ascontiguousarray(img[..., 0], dtype=np.float32)
    if not np.isfinite(d).all():
        raise ValueError("the picture contains NaN or infinite values")
    r, info = RL.relief(d, sigma_r, source, device=llf_device, tick=tick)
    return r, info


def run(path, level, budget_gib=6.5, out_dir=None, log=print, stage=None):
    """stage(name, done=None, total=None): the progress of each step; None = silent. Progress only."""
    if stage:
        stage("reading")
    source = source_for(path)
    img = photo_io.read_image_float(path)
    H, W = img.shape[:2]
    grey = file_is_grey(path)
    if grey is None:
        grey = float(max(np.abs(img[..., 0] - img[..., 1]).max(), np.abs(img[..., 0] - img[..., 2]).max())) <= GREY_TOL
    gpu = "cuda" if torch.cuda.is_available() else None
    cdev, ldev = devices_for(H, W, not grey, budget_gib, gpu)
    log(f"[relief] {os.path.basename(path)}: {W}x{H} | input '{source}' | strength {level} | "
        f"{'grey' if grey else 'colour'} (from the file) | devices {cdev or '-'} / {ldev}")
    out, info = relief_image(img, level, source, cdev, ldev, log=log, grey=grey, stage=stage)
    stem, ext = os.path.splitext(os.path.basename(path))
    d = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)           # a new --out folder (as the depth command does)
    # never over an existing file: <stem>_relief_strength<N>_vNNN<ext>; the earlier unversioned names (and the
    # pre-2026-10-31 `_relief_level<N>`) count as version 0
    kind = ext.lower().lstrip(".")
    if stage:
        stage("writing")
    dst = outnames.write_versioned(d, f"{stem}_relief_strength{int(level)}",
                                   [(kind, photo_io.encode_depth(out, kind, tagged=(source == RL.SOURCES[2])))],
                                   aliases=(f"{stem}_relief_level{int(level)}",))[0]
    if gpu:
        torch.cuda.empty_cache()
    return dst, info
