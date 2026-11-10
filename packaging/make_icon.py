r"""make_icon.py — writes vathography_app\assets\vathography_app.ico: an original serif "V" drawn from polygons (no
font: the earlier icon used Palatino Linotype's glyph, whose licence did not clearly
cover an app icon), text colour #eeebe6 on the window's black (#121110), the colours of gui.py. Roman V: thick left
stroke, thin right stroke, pointed foot, bracketed top serifs. Every size is drawn on its own (no downscale of the
256 px one) so the 16 / 24 / 32 px taskbar sizes stay sharp. Run once after a design change; the .ico is committed.
Usage: python packaging\make_icon.py [--preview <png>]
"""
import os
import sys

from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "vathography_app", "assets", "vathography_app.ico")
BG, FG = (0x12, 0x11, 0x10, 255), (0xee, 0xeb, 0xe6, 255)
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

# the glyph in a unit square (x right, y down); every shape is filled in FG
SERIF_TOP, STEM_TOP, FOOT_Y = 0.188, 0.205, 0.800
L_OUT, L_IN, R_IN, R_OUT = 0.243, 0.335, 0.715, 0.752      # stroke edges at STEM_TOP: thick left, thin right
FOOT_L, FOOT_R = 0.494, 0.506                                # the pointed foot (a hair wide so it survives 16 px)
L_IN_END, R_IN_END = 0.530, 0.488                            # where each inner edge would reach at FOOT_Y


def _x_at(x0, x1, y):                                       # an edge from (x0, STEM_TOP) to (x1, FOOT_Y), x at y
    return x0 + (x1 - x0) * (y - STEM_TOP) / (FOOT_Y - STEM_TOP)


def _shapes():
    # crotch = where the two inner edges meet
    a, b = (L_IN_END - L_IN), (R_IN_END - R_IN)
    t = (R_IN - L_IN) / (a - b)
    crotch = (L_IN + a * t, STEM_TOP + (FOOT_Y - STEM_TOP) * t)
    v = [(L_OUT, STEM_TOP), (L_IN, STEM_TOP), crotch, (R_IN, STEM_TOP), (R_OUT, STEM_TOP),
         (FOOT_R, FOOT_Y), (FOOT_L, FOOT_Y)]
    bars = [[(L_OUT - 0.055, SERIF_TOP), (L_IN + 0.050, SERIF_TOP), (L_IN + 0.050, STEM_TOP), (L_OUT - 0.055, STEM_TOP)],
            [(R_IN - 0.040, SERIF_TOP), (R_OUT + 0.045, SERIF_TOP), (R_OUT + 0.045, STEM_TOP), (R_IN - 0.040, STEM_TOP)]]
    h, d = 0.045, 0.030                                      # bracket: height down the stroke, reach along the bar
    y = STEM_TOP + h
    brackets = [[(L_OUT - d, STEM_TOP), (L_OUT, STEM_TOP), (_x_at(L_OUT, FOOT_L, y), y)],
                [(L_IN + d, STEM_TOP), (L_IN, STEM_TOP), (_x_at(L_IN, L_IN_END, y), y)],
                [(R_IN - d, STEM_TOP), (R_IN, STEM_TOP), (_x_at(R_IN, R_IN_END, y), y)],
                [(R_OUT + d, STEM_TOP), (R_OUT, STEM_TOP), (_x_at(R_OUT, FOOT_R, y), y)]]
    return [v] + bars + brackets


V_SHAPES = _shapes()


def draw(px):
    ss = 8                                    # drawn 8x then reduced once: smooth edges, no blur from a chain of resizes
    s = px * ss
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=round(s * 0.18), fill=BG)
    for poly in V_SHAPES:
        d.polygon([(x * s, y * s) for x, y in poly], fill=FG)
    return im.resize((px, px), Image.LANCZOS)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imgs = [draw(px) for px in SIZES]
    tmp = OUT + ".tmp"
    imgs[-1].save(tmp, format="ICO", sizes=[(p, p) for p in SIZES], append_images=imgs[:-1])
    os.replace(tmp, OUT)
    print(f"[icon] {OUT} sizes {SIZES}")
    if "--preview" in sys.argv:
        prev = Image.new("RGBA", (sum(SIZES) + 10 * len(SIZES), 256), (128, 128, 128, 255))
        x = 0
        for im in imgs:
            prev.paste(im, (x, 0), im)
            x += im.width + 10
        prev.save(sys.argv[sys.argv.index("--preview") + 1])


if __name__ == "__main__":
    main()
