r"""make_decoy_assets.py - rebuilds the two decoy files the app ships from their original, decoy_hand.png.

  vathography_app\engine\assets\decoy_hand_x3.5.npz   (tiles)
  vathography_app\engine\assets\decoy_hand_x2.5.npz   (the global pass)

The decoy is a near-object cue placed in a strip beside every model input: the hand photo scaled (cubic), blurred,
with a blurred elliptical alpha. The app loads the precomputed arrays, so it never needs OpenCV for this step.
Needs numpy + opencv-python (only to rebuild). Usage: python packaging\asset_originals\make_decoy_assets.py

Licence: AGPL-3.0-only with the additional permissions in LICENSE (this repository).
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "vathography_app", "engine", "assets")
SCALES = (3.5, 2.5)            # tiles, global pass
DECOY_BLUR = 6.0               # hand blur sigma
DECOY_ALPHA_BLUR = 8.0         # elliptical alpha blur sigma


def prepare_decoy(hand_bgr, scale):
    big = cv2.resize(hand_bgr, (int(hand_bgr.shape[1] * scale), int(hand_bgr.shape[0] * scale)),
                     interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:big.shape[0], 0:big.shape[1]]
    ell = (((xx - big.shape[1] / 2) / (big.shape[1] / 2)) ** 2
           + ((yy - big.shape[0] / 2) / (big.shape[0] / 2)) ** 2) <= 1.0
    return cv2.GaussianBlur(big, (0, 0), DECOY_BLUR), cv2.GaussianBlur(ell.astype(np.float32), (0, 0), DECOY_ALPHA_BLUR)


def main():
    hand = cv2.imread(os.path.join(HERE, "decoy_hand.png"), cv2.IMREAD_COLOR)
    assert hand is not None and hand.dtype == np.uint8 and hand.shape == (195, 145, 3), "decoy_hand.png: 145 x 195 RGB"
    for scale in SCALES:
        obj, alpha = prepare_decoy(hand, scale)
        out = os.path.join(OUT, f"decoy_hand_x{scale:g}.npz")
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, obj=obj, alpha=alpha, scale=np.float64(scale))
        os.replace(tmp, out)
        print(f"x{scale:g}: obj {obj.shape} alpha {alpha.shape} -> {out}")


if __name__ == "__main__":
    main()
