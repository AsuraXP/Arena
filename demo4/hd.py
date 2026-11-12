#!/usr/bin/env python3
"""
hd.py — the hero shot, rebuilt with actual animation craft.

Same scene as demo3's street shot, but with:
  * 3 depth layers + depth-of-field blur
  * anticipation / overshoot easing on the arm raise
  * overlapping action: shoulder leads, forearm lags 4 frames, hand lags 6
  * weight shift, breathing, head tilt (secondary motion)
  * rim light, projected figure shadow, volumetric light cone
  * dust motes lit only inside the beam
  * 3-sample motion blur on the moving parts
  * colour grade (lifted blue shadows, warm highlights, S-curve)
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.abspath(__file__))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
DUR = 8.0
F_BIG = os.path.join(ROOT, "../demo/fonts/Anton.ttf")
F_SUB = os.path.join(ROOT, "../demo/fonts/Inter-SemiBold.ttf")

# world units == screen px at z=1, origin at centre
WIN = (-150, -330, 130, -70)


# ------------------------------------------------------------------ easing
def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out_back(x, s=1.9):
    x = clamp01(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def ease_out_cubic(x):
    return 1 - (1 - clamp01(x)) ** 3


def ease_in_out(x):
    x = clamp01(x)
    return 3 * x * x - 2 * x * x * x


def lerp(a, b, k):
    return a + (b - a) * k


def hexc(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------- glow
_GLOW = {}


def glow_sprite(size, color):
    k = (size, tuple(color))
    if k not in _GLOW:
        y, x = np.mgrid[0:size, 0:size]
        c = (size - 1) / 2.0
        d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
        _GLOW[k] = ((np.clip(1 - d, 0, 1) ** 2.4))[:, :, None] * np.array(color)[None, None, :]
    return _GLOW[k]


def add_glow(arr, cx, cy, size, color, strength):
    g = glow_sprite(int(size), color)
    h, w = g.shape[:2]
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, arr.shape[1]), min(y0 + h, arr.shape[0])
    if xb <= xa or yb <= ya:
        return
    arr[ya:yb, xa:xb] += g[ya - y0:ya - y0 + (yb - ya),
                          xa - x0:xa - x0 + (xb - xa)] * strength


# -------------------------------------------------------------------- rig
def limb(d, o, a1, l1, a2, l2, w, col):
    j = (o[0] + math.sin(a1) * l1, o[1] + math.cos(a1) * l1)
    e = (j[0] + math.sin(a2) * l2, j[1] + math.cos(a2) * l2)
    d.line([o, j], fill=col, width=max(1, int(w)))
    d.line([j, e], fill=col, width=max(1, int(w * 0.9)))
    r = w * 0.5
    d.ellipse([j[0] - r, j[1] - r, j[0] + r, j[1] + r], fill=col)
    r2 = w * 0.46
    d.ellipse([e[0] - r2, e[1] - r2, e[0] + r2, e[1] + r2], fill=col)
    return e


def draw_person(d, x, ground_y, h, pose, col=(8, 6, 5, 255)):
    head_r = h * 0.082
    spine = h * 0.40
    hip_w = h * 0.075
    lean = pose.get("lean", 0.0)
    hx = x
    hy = ground_y - h * 0.47
    sx = hx + math.sin(lean) * spine
    sy = hy - math.cos(lean) * spine
    hdx = sx + math.sin(lean + pose.get("head", 0)) * head_r * 2.0
    hdy = sy - math.cos(lean + pose.get("head", 0)) * head_r * 2.0

    limb(d, (hx - hip_w * 0.55, hy), pose.get("thigh_far", 0.05), h * 0.24,
         pose.get("shin_far", 0.05), h * 0.235, h * 0.082, col)
    limb(d, (sx - hip_w * 0.75, sy + h * 0.03), pose.get("arm_far", 0.14), h * 0.19,
         pose.get("fore_far", 0.10), h * 0.175, h * 0.062, col)
    d.polygon([(hx - hip_w, hy + 4), (hx + hip_w, hy + 4),
               (sx + hip_w * 1.55, sy), (sx - hip_w * 1.55, sy)], fill=col)
    r = hip_w * 1.35
    d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=col)
    r2 = hip_w * 0.95
    d.ellipse([hx - r2, hy - r2, hx + r2, hy + r2], fill=col)
    limb(d, (hx + hip_w * 0.55, hy), pose.get("thigh", 0.05), h * 0.24,
         pose.get("shin", 0.05), h * 0.235, h * 0.086, col)
    limb(d, (sx + hip_w * 0.75, sy + h * 0.03), pose.get("arm", 0.14), h * 0.19,
         pose.get("fore", 0.10), h * 0.175, h * 0.066, col)
    d.line([(sx, sy - hip_w * 0.4), (hdx, hdy)], fill=col, width=max(1, int(h * 0.05)))
    d.ellipse([hdx - head_r, hdy - head_r, hdx + head_r, hdy + head_r], fill=col)


# ------------------------------------------------------------------- pose
def wave_pose(t):
    """
    Overlapping action: the shoulder starts moving first, the forearm follows
    a few frames later, the whole thing settles with ease-out-back overshoot.
    """
    RAISE_AT, RAISE_LEN = 0.55, 0.85
    k = clamp01((t - RAISE_AT) / RAISE_LEN)
    raise_k = ease_out_back(k, 1.5)            # overshoot then settle

    # shoulder phase and the lagging forearm phase
    sw_s = math.sin((t - RAISE_AT) * 2 * math.pi * 1.35)
    sw_f = math.sin((t - RAISE_AT - 0.13) * 2 * math.pi * 1.35)   # ~4 frame lag
    sw_h = math.sin((t - RAISE_AT - 0.20) * 2 * math.pi * 1.35)

    arm_down, arm_up = 0.16, 2.30
    fore_down, fore_up = 0.10, 0.34

    arm = lerp(arm_down, arm_up + sw_s * 0.13, raise_k)
    fore = lerp(fore_down, fore_up + sw_f * 0.55, raise_k) + sw_h * 0.10 * raise_k
    # counter-rotation in the torso from the arm swing (secondary motion)
    lean = -0.03 * math.sin(t * 1.15) + sw_f * 0.025 * raise_k
    head = 0.05 * math.sin(t * 0.95) - sw_f * 0.05 * raise_k
    # weight shift between feet
    thigh = 0.05 + 0.03 * math.sin(t * 0.7)
    thigh_far = -0.05 - 0.03 * math.sin(t * 0.7)
    breathe = 0.012 * math.sin(t * 1.5)
    return {"arm": arm, "fore": fore, "arm_far": 0.15, "fore_far": 0.12,
            "lean": lean + breathe, "head": head,
            "thigh": thigh, "shin": 0.03, "thigh_far": thigh_far, "shin_far": 0.02}


# ------------------------------------------------------------------ scene
class HeroShot:
    def __init__(self):
        self.rng = np.random.default_rng(4)
        r = np.random.default_rng(11)
        # background: distant buildings, heavily blurred
        bg = Image.new("RGB", (W, H), (0, 0, 0))
        bd = ImageDraw.Draw(bg)
        y_horizon = H * 0.74
        x = -200
        while x < W + 200:
            bw = r.uniform(90, 210)
            bh = r.uniform(120, 330)
            bd.rectangle([x, y_horizon - bh, x + bw, y_horizon], fill=(15, 22, 40))
            x += bw + r.uniform(12, 46)
        self.bg = bg.filter(ImageFilter.GaussianBlur(7))

        self.bricks = [(r.uniform(-1300, 1300), r.uniform(-560, 130),
                        r.uniform(0.5, 1.0)) for _ in range(260)]
        self.motes = [(r.uniform(-260, 240), r.uniform(-330, 60),
                       r.uniform(0.4, 1.0), r.uniform(0.6, 2.2)) for _ in range(70)]
        self.stars = [(r.uniform(-1200, 1200), r.uniform(-560, -260),
                       r.uniform(0.3, 1.0)) for _ in range(90)]

    # -- helpers -------------------------------------------------------
    SCALE = 2.05          # world units -> screen px (this is the shot's framing)
    ORIGIN = (-10, -200)  # world point that lands at screen centre (= the window)

    def wx(self, x):
        return W / 2 + (x - self.ORIGIN[0]) * self.SCALE + self.drift_x

    def wy(self, y):
        return H / 2 + (y - self.ORIGIN[1]) * self.SCALE + self.drift_y

    def person_layer(self, t):
        """figure on transparent RGBA: projected shadow, rim light, dark body"""
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        wx0, wy0, wx1, wy1 = WIN
        cx = (wx0 + wx1) / 2 - 8
        base = wy1 + 16
        h = (wy1 - wy0) * 0.94
        pose = wave_pose(t)
        sx, sy = self.wx(cx), self.wy(base)

        # 1. projected shadow on the wall behind (offset, blurred)
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        draw_person(sd, self.wx(cx + 30), self.wy(base + 6), h * 1.03, pose,
                    (92, 54, 24, 130))
        sh = sh.filter(ImageFilter.GaussianBlur(6))
        lay = Image.alpha_composite(lay, sh)

        # 2. rim light: a warm silhouette nudged up-left, then the body on top
        rim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rim)
        draw_person(rd, sx - 5, sy - 5, h, pose, (255, 202, 138, 235))
        lay = Image.alpha_composite(lay, rim)

        body = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(body)
        draw_person(bd, sx, sy, h, pose, (16, 10, 7, 255))
        lay = Image.alpha_composite(lay, body)
        return lay

    # -- frame ---------------------------------------------------------
    def render(self, t, frame_i=0):
        self.drift_x = math.sin(t * 0.22) * 3.0
        self.drift_y = math.cos(t * 0.19) * 2.0

        # slow push-in
        z = 1.04 + 0.16 * ease_in_out(t / DUR)
        arr = np.asarray(self.bg).astype(np.float32)

        # stars + moon glow (background layer)
        for (sx, sy, sb) in self.stars:
            add_glow(arr, self.wx(sx), self.wy(sy), 8, (190, 210, 245), 0.45 * sb)
        add_glow(arr, self.wx(-560), self.wy(-420), 520, (90, 120, 180), 0.22)

        # ---------------- mid layer: facade --------------------------
        mid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        md = ImageDraw.Draw(mid)
        fx0, fy0, fx1, fy1 = self.wx(-1400), self.wy(-900), self.wx(1400), self.wy(120)
        md.rectangle([fx0, fy0, fx1, fy1], fill=hexc("20242f"))
        for (bx, by, bs) in self.bricks:
            md.rectangle([self.wx(bx), self.wy(by), self.wx(bx) + 42 * bs * self.SCALE,
                          self.wy(by) + 13 * self.SCALE], fill=hexc("242833"))
        # horizontal banding for scale
        for i in range(9):
            yy = self.wy(-900 + i * 140)
            md.line([(fx0, yy), (fx1, yy)], fill=hexc("1b1f29"), width=3)

        # unlit windows with sills, each with a faint interior gradient
        for wx in (-940, -600, 360, 700, 1040):
            for wy in (-540, -350, -160):
                x0, y0 = self.wx(wx), self.wy(wy)
                x1, y1 = self.wx(wx + 120), self.wy(wy + 150)
                md.rectangle([x0 - 12, y0 - 12, x1 + 12, y1 + 12], fill=hexc("343a48"))
                md.rectangle([x0, y0, x1, y1], fill=hexc("0a0d15"))
                md.rectangle([x0, y0, x1, y0 + (y1 - y0) * 0.35], fill=hexc("0d1119"))
                md.rectangle([x0 - 20, y1 + 12, x1 + 20, y1 + 28], fill=hexc("3c4250"))
        mid_a = np.asarray(mid).astype(np.float32)
        mid_al = mid_a[:, :, 3:4] / 255.0
        # facade shadow (unlit parts) sits under everything else
        arr = arr * (1 - mid_al) + mid_a[:, :, :3] * mid_al

        # the lit window + interior
        wx0, wy0, wx1, wy1 = WIN
        lit = Image.new("RGB", (W, H), (0, 0, 0))
        ld = ImageDraw.Draw(lit)
        glow_k = 0.965 + 0.035 * math.sin(t * 1.35)
        x0, y0, x1, y1 = self.wx(wx0), self.wy(wy0), self.wx(wx1), self.wy(wy1)
        ld.rectangle([x0 - 14, y0 - 14, x1 + 14, y1 + 14], fill=hexc("3a4050"))
        # interior wall: warm gradient, brighter near the lamp
        for i in range(int(y1 - y0)):
            k = i / max(1, (y1 - y0))
            c = tuple(int(v * glow_k * (1.06 - 0.26 * k)) for v in (255, 208, 152))
            ld.line([(x0, y0 + i), (x1, y0 + i)], fill=c)
        lit_a = np.asarray(lit).astype(np.float32)
        # paste interior only inside the window rect
        mask = np.zeros((H, W, 1), np.float32)
        mask[int(y0):int(y1), int(x0):int(x1)] = 1.0
        arr = arr * (1 - mask) + lit_a * mask

        # figure layer (shadow + rim + body)
        fig = self.person_layer(t)
        fa = np.asarray(fig).astype(np.float32)
        alpha = fa[:, :, 3:4] / 255.0
        arr = arr * (1 - alpha) + fa[:, :, :3] * alpha

        # mullions on top of the figure
        mull = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        mu = ImageDraw.Draw(mull)
        mc = tuple(int(v * glow_k * 0.62) for v in (240, 190, 132)) + (255,)
        mu.rectangle([self.wx((wx0 + wx1) / 2) - 4, y0, self.wx((wx0 + wx1) / 2) + 4, y1],
                     fill=mc)
        mu.rectangle([x0, self.wy(wy0 + 40), x1, self.wy(wy0 + 48)], fill=mc)
        ma = np.asarray(mull).astype(np.float32)
        m_al = ma[:, :, 3:4] / 255.0
        arr = arr * (1 - m_al) + ma[:, :, :3] * m_al

        # ---------------- light spill -------------------------------
        cxw, cyw = self.wx((wx0 + wx1) / 2), self.wy(wy0)
        add_glow(arr, cxw, self.wy((wy0 + wy1) / 2), 1500, (255, 168, 92), 0.40 * glow_k)
        # volumetric cone onto the street
        cone = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cone)
        cd.polygon([(x0, y1), (x1, y1),
                    (self.wx(( wx0 + wx1) / 2 + 620), self.wy(700)),
                    (self.wx((wx0 + wx1) / 2 - 620), self.wy(700))],
                   fill=(255, 176, 104, 26))
        cone = cone.filter(ImageFilter.GaussianBlur(26))
        ca = np.asarray(cone).astype(np.float32)
        c_al = ca[:, :, 3:4] / 255.0
        arr = arr + ca[:, :, :3] * c_al * 0.55

        # lit pool + wet reflection on the pavement
        add_glow(arr, cxw, self.wy(300), 1300, (255, 156, 82), 0.26 * glow_k)
        for k in range(8):
            add_glow(arr, cxw + math.sin(k * 1.9 + t * 0.9) * 9,
                     self.wy(250 + k * 34), 260 - k * 22, (255, 150, 74),
                     (0.15 - k * 0.015) * glow_k)

        # dust motes, only bright inside the beam
        for (mx, my, mb, sp) in self.motes:
            yy = my + math.sin(t * sp + mx * 0.05) * 6
            xx = mx + math.sin(t * sp * 0.7 + my * 0.03) * 5
            rr = 1.0
            sxp, syp = self.wx(xx), self.wy(yy)
            in_beam = 1.0 - min(1.0, abs(sxp - cxw) / (W * 0.30))
            if in_beam > 0.15:
                add_glow(arr, sxp, syp, 12, (255, 226, 186), 0.5 * mb * in_beam)

        # ---------------- foreground (out of focus) -----------------
        fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fd = ImageDraw.Draw(fg)
        ink = (3, 4, 7, 255)
        def box(x0, y0, x1, y1, fill):
            fd.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill)
        box(-40, -40, self.wx(-1150), H + 40, ink)
        box(self.wx(1140), -40, W + 40, H + 40, ink)
        box(-40, -40, W + 40, self.wy(-650), ink)
        box(-40, self.wy(500), W + 40, H + 40, ink)
        fg = fg.filter(ImageFilter.GaussianBlur(11))
        fga = np.asarray(fg).astype(np.float32)
        f_al = fga[:, :, 3:4] / 255.0
        arr = arr * (1 - f_al) + fga[:, :, :3] * f_al

        # ---------------- grade -------------------------------------
        # vignette
        y, x = np.mgrid[0:H, 0:W]
        d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        arr = arr * np.clip(1 - d ** 4.2 * 0.92, 0.28, 1)[:, :, None]
        # lift shadows toward blue, warm the highlights
        a = np.clip(arr / 255.0, 0, 1)
        a = np.clip((a - 0.5) * 1.10 + 0.5, 0, 1) ** 1.03          # S-curve contrast
        shadow_lift = np.array([0.020, 0.030, 0.058])
        hi_warm = np.array([0.020, 0.010, -0.012])
        a = a + shadow_lift * (1 - a) ** 2.4 + hi_warm * (a ** 2.2)
        a = np.clip(a, 0, 1)
        arr = a * 255.0
        # grain
        arr = arr + self.rng.normal(0, 3.4, arr.shape)
        return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- caption
def caption(frame, text, size=46, big=False):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(F_BIG if big else F_SUB, 96 if big else size)
    fs = 96 if big else size
    tw = d.textlength(text, font=font)
    x, y = (W - tw) / 2, H - 150
    d.text((x, y), text, font=font, fill=(0, 0, 0), stroke_width=6, stroke_fill=(0, 0, 0))
    d.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    sc = HeroShot()
    n = int(DUR * FPS)
    tmp = os.path.join(ROOT, "hd_raw.mp4")
    p = subprocess.Popen([FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "slow", "-crf", "19",
                          "-pix_fmt", "yuv420p", "-an", tmp],
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # motion blur: average 3 temporal samples of the moving figure
    for i in range(n):
        t = i / FPS
        acc = None
        for s in (-1, 0, 1):
            fr = sc.render(t + s * (0.5 / FPS)).astype(np.float32)
            acc = fr if acc is None else acc + fr
        fr = (acc / 3.0)
        fr = caption(np.clip(fr, 0, 255).astype(np.uint8), "Someone is waving at you.")
        p.stdin.write(fr.tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{n}", flush=True)
    p.stdin.close()
    err = p.stderr.read().decode()[-800:]
    if p.wait() != 0:
        sys.exit("encode failed: " + err)
    print("wrote", tmp)


if __name__ == "__main__":
    main()
