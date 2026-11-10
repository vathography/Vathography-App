# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
# Based on the Fast Bilateral Solver reference implementation, Copyright (c) 2016 Ben Poole, MIT License (see NOTICE.txt).
r"""sky_fbs.py — Fast Bilateral Solver (Barron & Poole, ECCV 2016), numpy + scipy, used by sky_repair.py.

A modified version of the authors' reference implementation, github.com/poolio/bilateral_solver (MIT, Ben Poole 2016;
the licence text is in NOTICE.txt). Changes from it: canvas-absolute spatial coordinates, a mixed-radix coordinate hash
(the reference's fixed radix collides on large canvases), guards against division by zero, scipy's `rtol` keyword, a
float32 output, and a spatial bilinear splat + slice (`interp="bilinear"`) that removes the square pattern of the
nearest-vertex slice.
Guide = uint8 RGB [H,W,3]; target and confidence = float32 [H,W]; output float32 [H,W].
"""
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import cg

RGB_TO_YUV = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)

TILE, OVERLAP, MARGIN = 2048, 128, 256               # fbs_tiled: memory follows one solve, not the whole band

DEFAULT_CG_TOL = 1e-5
DEFAULT_CG_MAXITER = 25
DEFAULT_A_DIAG_MIN = 1e-5
BISTOCHASTIZE_ITERS = 10


def rgb2yuv(im):
    return np.tensordot(im.astype(np.float64), RGB_TO_YUV, ([2], [1])) + YUV_OFFSET


class BilateralGrid:
    """Hashed bilateral grid over (x/σs, y/σs, Y/σl, U/σc, V/σc)."""

    def __init__(self, im_rgb_u8, sigma_spatial=8.0, sigma_luma=8.0, sigma_chroma=8.0, origin=(0, 0),
                 interp="nearest"):
        if im_rgb_u8.ndim != 3 or im_rgb_u8.shape[2] < 3:
            raise ValueError("guide must be [H,W,3] uint8 RGB")
        if interp not in ("nearest", "bilinear"):
            raise ValueError(f"interp must be 'nearest' or 'bilinear', got {interp!r}")
        self.interp = interp
        im_yuv = rgb2yuv(im_rgb_u8[..., :3])
        h, w = im_rgb_u8.shape[:2]
        ox, oy = int(origin[0]), int(origin[1])
        Iy, Ix = np.mgrid[:h, :w]
        fx = (Ix + ox) / float(sigma_spatial)   # canvas-absolute fractional grid coordinates
        fy = (Iy + oy) / float(sigma_spatial)
        x_coords = fx.astype(np.int64)
        y_coords = fy.astype(np.int64)
        l_coords = (im_yuv[..., 0] / float(sigma_luma)).astype(np.int64)
        c_coords = (im_yuv[..., 1:] / float(sigma_chroma)).astype(np.int64)
        coords = np.dstack((x_coords, y_coords, l_coords, c_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.dim = coords_flat.shape[1]
        self.npixels = coords_flat.shape[0]
        self.shape = (h, w)
        if interp == "nearest":
            self.mins = coords_flat.min(0) - 1
            spans = coords_flat.max(0) - self.mins + 2
            self.strides = np.cumprod(np.concatenate([[1], spans[:-1]])).astype(np.int64)
            self.spans = spans
            self._compute_factorization(coords_flat)
        else:
            # bilinear candidates can reach one cell below/above the nearest cell -> widen the hash range
            self.mins = coords_flat.min(0) - 2
            spans = coords_flat.max(0) - self.mins + 3
            self.strides = np.cumprod(np.concatenate([[1], spans[:-1]])).astype(np.int64)
            self.spans = spans
            fxf = fx.reshape(-1)
            fyf = fy.reshape(-1)
            del Ix, Iy, fx, fy, coords, im_yuv, x_coords, y_coords, l_coords, c_coords  # memory: free before the 4N build
            self._compute_factorization_bilinear(coords_flat, fxf, fyf)

    def _hash(self, coords):
        return ((coords - self.mins) * self.strides).sum(1)

    def _unhash(self, hashes):
        """Inverse of _hash (mixed radix) -> int64 coords [n, dim]."""
        out = np.empty((hashes.shape[0], self.dim), np.int64)
        rem = hashes.copy()
        for d in range(self.dim - 1, -1, -1):
            q = rem // self.strides[d]
            out[:, d] = q + self.mins[d]
            rem = rem - q * self.strides[d]
        return out

    def _build_blur(self, unique_hashes, unique_coords):
        blur = 2.0 * self.dim * identity(self.nvertices, format="csr")
        for d in range(self.dim):
            for offset in (-1, 1):
                nb = unique_coords.copy()
                nb[:, d] += offset
                nb_hash = self._hash(nb)
                pos = np.searchsorted(unique_hashes, nb_hash)
                pos[pos >= self.nvertices] = 0
                valid = unique_hashes[pos] == nb_hash
                rows = np.nonzero(valid)[0]
                blur = blur + csr_matrix((np.ones(len(rows)), (rows, pos[valid])),
                                         shape=(self.nvertices, self.nvertices))
        self.B = blur.tocsr()

    def _compute_factorization_bilinear(self, coords_flat, fx, fy):
        """Deviation 8: centre-based spatial bilinear splat. bx = floor(fx - 0.5), tx = fx - 0.5 - bx;
        the 4 candidate cells (bx+{0,1}, by+{0,1}) carry the pixel's OWN colour bin."""
        n = self.npixels
        bx = np.floor(fx - 0.5).astype(np.int64)
        by = np.floor(fy - 0.5).astype(np.int64)
        tx = (fx - 0.5 - bx)
        ty = (fy - 0.5 - by)
        # colour part of the hash is shared by the 4 candidates: compute it once (memory: no n x 5 temps)
        colour_hash = ((coords_flat[:, 2:] - self.mins[2:]) * self.strides[2:]).sum(1)
        hashes = []
        weights = []
        for dx, dy, wgt in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
            hashes.append((bx + dx - self.mins[0]) * self.strides[0] + (by + dy - self.mins[1]) * self.strides[1] + colour_hash)
            weights.append(wgt)
        del colour_hash, tx, ty, bx, by
        hashed_all = np.concatenate(hashes)
        del hashes
        w_all = np.concatenate(weights)
        del weights
        pix_all = np.tile(np.arange(n, dtype=np.int32 if n < 2**31 - 1 else np.int64), 4)
        # a pixel exactly at a cell centre has weight 0 on its +1 neighbours: drop those candidates so no
        # zero-mass ghost vertex enters the grid (they would receive no data and skew bistochastisation)
        keep = w_all > 0.0
        if not keep.all():
            hashed_all = hashed_all[keep]
            w_all = w_all[keep]
            pix_all = pix_all[keep]
        del keep
        unique_hashes, inv = np.unique(hashed_all, return_inverse=True)
        del hashed_all
        self.nvertices = len(unique_hashes)
        self.S = csr_matrix((w_all, (inv, pix_all)), shape=(self.nvertices, n))
        unique_coords = self._unhash(unique_hashes)
        self._build_blur(unique_hashes, unique_coords)

    def _compute_factorization(self, coords_flat):
        hashed = self._hash(coords_flat)
        unique_hashes, unique_idx, idx = np.unique(hashed, return_index=True, return_inverse=True)
        self.nvertices = len(unique_hashes)
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))),
                            shape=(self.nvertices, self.npixels))
        unique_coords = coords_flat[unique_idx]
        blur = 2.0 * self.dim * identity(self.nvertices, format="csr")
        for d in range(self.dim):
            for offset in (-1, 1):
                nb = unique_coords.copy()
                nb[:, d] += offset
                nb_hash = self._hash(nb)
                pos = np.searchsorted(unique_hashes, nb_hash)
                pos[pos >= self.nvertices] = 0
                valid = unique_hashes[pos] == nb_hash
                rows = np.nonzero(valid)[0]
                blur = blur + csr_matrix((np.ones(len(rows)), (rows, pos[valid])),
                                         shape=(self.nvertices, self.nvertices))
        self.B = blur.tocsr()

    def splat(self, x):
        return self.S.dot(x)

    def slice(self, y):
        return self.S.T.dot(y)

    def blur(self, x):
        return self.B.dot(x)


def bistochastize(grid, maxiter=BISTOCHASTIZE_ITERS, correct_m=True):
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for _ in range(maxiter):
        n = np.sqrt(n * m / np.maximum(grid.blur(n), 1e-12))
    if correct_m:
        # reference: "Correct m to satisfy the assumption of bistochastization
        # regardless of how many iterations have been run."
        m = n * grid.blur(n)
    return diags(n, 0), diags(m, 0)


class BilateralSolver:
    def __init__(self, grid, lam=64.0, A_diag_min=DEFAULT_A_DIAG_MIN, cg_tol=DEFAULT_CG_TOL,
                 cg_maxiter=DEFAULT_CG_MAXITER, correct_m=True):
        self.grid = grid
        self.lam = float(lam)
        self.A_diag_min = A_diag_min
        self.cg_tol = cg_tol
        self.cg_maxiter = int(cg_maxiter)
        Dn, Dm = bistochastize(grid, correct_m=correct_m)
        self.A_smooth = (Dm - Dn.dot(grid.B).dot(Dn)).tocsr()

    def solve(self, target, confidence):
        x = target.reshape(-1).astype(np.float64)
        w = confidence.reshape(-1).astype(np.float64)
        w_splat = self.grid.splat(w)
        A = (self.lam * self.A_smooth + diags(w_splat, 0)).tocsr()
        b = self.grid.splat(x * w)
        A_diag = np.maximum(A.diagonal(), self.A_diag_min)
        M = diags(1.0 / A_diag, 0)
        y0 = b / np.maximum(w_splat, 1e-12)
        yhat, info = cg(A, b, x0=y0, M=M, rtol=self.cg_tol, maxiter=self.cg_maxiter)
        bn = float(np.linalg.norm(b))
        residual = float(np.linalg.norm(A.dot(yhat) - b) / bn) if bn > 0 else 0.0
        xhat = self.grid.slice(yhat)
        return xhat.reshape(target.shape).astype(np.float32), int(info), residual


def fbs_filter(guide_rgb_u8, target, confidence, sigma_spatial=8.0, sigma_luma=8.0, sigma_chroma=8.0,
               lam=64.0, cg_maxiter=DEFAULT_CG_MAXITER, cg_tol=DEFAULT_CG_TOL, origin=(0, 0),
               correct_m=True, interp="nearest"):
    """One solve. Returns (out float32 [H,W], info dict)."""
    grid = BilateralGrid(guide_rgb_u8, sigma_spatial, sigma_luma, sigma_chroma, origin=origin, interp=interp)
    solver = BilateralSolver(grid, lam=lam, cg_maxiter=cg_maxiter, cg_tol=cg_tol, correct_m=correct_m)
    out, info, residual = solver.solve(target, confidence)
    return out, dict(nvertices=grid.nvertices, npixels=grid.npixels, cg_info=info,
                     residual=residual, vertex_ratio=grid.nvertices / max(grid.npixels, 1), interp=interp)


def _starts(n):
    if n <= TILE:
        return [0]
    k = int(np.ceil((n - OVERLAP) / (TILE - OVERLAP)))
    return [round(i * (n - TILE) / (k - 1)) for i in range(k)]


def _ramp(a, b, n):
    w = np.minimum(np.minimum(np.arange(b - a) + 1, np.arange(b - a)[::-1] + 1) / OVERLAP, 1).astype(np.float32)
    if a == 0:
        w[:OVERLAP] = 1
    if b == n:
        w[-OVERLAP:] = 1
    return w


def fbs_tiled(guide_rgb_u8, target, confidence, wanted, sigma_spatial=8.0, sigma_luma=8.0, sigma_chroma=8.0, lam=64.0,
              interp="bilinear"):
    """fbs_filter in tiles of TILE px, each solved with MARGIN px of context on the canvas's own grid, cross-faded over
    OVERLAP px. Only tiles holding a `wanted` pixel are solved; the others keep the target. Returns (out, tiles solved)."""
    h, w = target.shape
    num = np.zeros((h, w), np.float32); den = np.zeros((h, w), np.float32)
    n = 0
    for y0 in _starts(h):
        for x0 in _starts(w):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            if not wanted[y0:y1, x0:x1].any():
                continue
            ya, xa, yb, xb = max(0, y0 - MARGIN), max(0, x0 - MARGIN), min(h, y1 + MARGIN), min(w, x1 + MARGIN)
            out = fbs_filter(guide_rgb_u8[ya:yb, xa:xb], target[ya:yb, xa:xb], confidence[ya:yb, xa:xb], sigma_spatial, sigma_luma,
                             sigma_chroma, lam, origin=(xa, ya), interp=interp)[0]
            wt = _ramp(y0, y1, h)[:, None] * _ramp(x0, x1, w)[None, :]
            num[y0:y1, x0:x1] += wt * out[y0 - ya:y1 - ya, x0 - xa:x1 - xa]; den[y0:y1, x0:x1] += wt
            n += 1
    return np.where(den > 0, num / np.maximum(den, 1e-6), target).astype(np.float32), n
