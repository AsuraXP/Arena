#!/usr/bin/env python3
"""
build2.py - "What if you never needed to sleep?" - animated build.

Same script as the tutorial's Opus version. Differences from demo/build.py:
three shots are *procedurally animated* frame-by-frame in PIL instead of
being still images with camera moves:

  * SCENE wave      - figure in the far window waves; body sways; light breathes
  * SCENE windows   - a whole skyline's windows go dark one at a time, timed
  * SCENE light-out - the payoff: the waving figure's window dims to black

Everything else (voice, stills, captions, SFX, mix, encode) is the same
$0 pipeline: no ElevenLabs, no paid agent.
"""
import json
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.abspath(__file__))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
SS = 1.1                       # render scale, gives room for camera drift
RW, RH = int(W * SS), int(H * SS)
GAP = 0.35

F_BIG = os.path.join(ROOT, "../demo/fonts/Anton.ttf")
F_SUB = os.path.join(ROOT, "../demo/fonts/Inter-SemiBold.ttf")
STILLS_DIRS = [os.path.join(ROOT, "shots"), os.path.join(ROOT, "../demo/shots")]
STILLS = STILLS_DIRS[0]


def still_path(name):
    for base in STILLS_DIRS:
        p = os.path.join(base, name + ".png")
        if os.path.exists(p):
            return p
    sys.exit("missing still: " + name)

INK = (7, 10, 18)
AMBER = (255, 186, 92)
AMBER_DIM = (196, 132, 58)
CYAN = (95, 208, 224)


# ----------------------------------------------------------------- helpers
def hexc(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def vgrad(w, h, top, bot):
    """vertical gradient as float array (h,w,3)"""
    t = np.linspace(0, 1, h)[:, None]
    arr = np.array(top)[None, :] * (1 - t) + np.array(bot)[None, :] * t
    return np.repeat(arr[:, None, :], w, axis=1)


def glow_sprite(size, color, power=2.2):
    """radial falloff sprite, RGB float array"""
    y, x = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
    a = np.clip(1 - d, 0, 1) ** power
    return a[:, :, None] * np.array(color)[None, None, :]


GLOW_CACHE = {}


def glow(size, color):
    k = (size, tuple(color))
    if k not in GLOW_CACHE:
        GLOW_CACHE[k] = glow_sprite(size, color)
    return GLOW_CACHE[k]


def add_glow(arr, cx, cy, size, color, strength):
    """additive radial glow into float array"""
    g = glow(int(size), color)
    h, w = g.shape[:2]
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, arr.shape[1]), min(y0 + h, arr.shape[0])
    if xb <= xa or yb <= ya:
        return
    gx0, gy0 = xa - x0, ya - y0
    sub = g[gy0:gy0 + (yb - ya), gx0:gx0 + (xb - xa)]
    arr[ya:yb, xa:xb] += sub * strength


def vignette_mask(w, h, p=4.6):
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    d = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2) / math.sqrt(2)
    return (np.clip(1 - d ** p * 0.95, 0.25, 1))[:, :, None]


VIG = vignette_mask(W, H)


def finish(arr, rng, grain=4.5):
    """crop to frame, vignette, grain -> uint8 RGB"""
    arr = arr[:H, :W]
    arr = arr * VIG
    if grain:
        arr = arr + rng.normal(0, grain, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- character
def draw_figure(d, cx, base_y, scale, arm_deg, lean=0.0, color=(6, 8, 14)):
    """flat silhouette bust with one waving arm; scale = head radius"""
    hr = scale
    sh_y = base_y - hr * 2.55
    d.ellipse([cx - hr + lean, sh_y - hr * 2.2, cx + hr + lean, sh_y + hr * 0.1], fill=color)
    # shoulders / torso
    d.polygon([(cx - hr * 1.85 + lean, base_y),
               (cx - hr * 1.55 + lean, sh_y + hr * 0.25),
               (cx + hr * 1.55 + lean, sh_y + hr * 0.25),
               (cx + hr * 1.85 + lean, base_y)], fill=color)
    # waving arm: shoulder pivot, rotate outward
    a = math.radians(arm_deg)
    sl = hr * 2.0
    sx, sy = cx + hr * 1.35 + lean, sh_y + hr * 0.7
    ex = sx + math.sin(a) * sl
    ey = sy - math.cos(a) * sl
    d.line([(sx, sy), (ex, ey)], fill=color, width=max(2, int(hr * 0.55)))
    # hand
    d.ellipse([ex - hr * 0.42, ey - hr * 0.42, ex + hr * 0.42, ey + hr * 0.42], fill=color)


# ----------------------------------------------------------------- scenes

class PlazaScene:
    """The generated street art, with a figure that actually waves inside the lit window."""

    # window rect in fractions of the source image (measured from the art)
    WIN = (0.616, 0.300, 0.686, 0.535)

    def __init__(self, duration, act_from=0.15, blackout=False):
        self.dur = duration
        self.act_from = act_from
        self.blackout = blackout
        art = Image.open(still_path("shot6")).convert("RGB")
        self.art = art.resize((RW, RH), Image.LANCZOS)
        a = np.asarray(self.art).astype(np.float32)
        self.base = a
        self.wall = a.copy()          # with the figure painted out, built lazily
        self._built = False

    def _build_clean(self):
        """replace the existing arm/hand silhouette with flat window light"""
        a = self.base.copy()
        h, w = a.shape[:2]
        x0, y0, x1, y1 = self.WIN
        px0, px1 = int(x0 * w), int(x1 * w)
        py0, py1 = int(y0 * h), int(y1 * h)
        # sample window light colour from an empty part of the window
        patch = a[py0 + int((py1-py0)*0.15):py0 + int((py1-py0)*0.35),
                  px0 + int((px1-px0)*0.10):px0 + int((px1-px0)*0.35)]
        light = patch.reshape(-1, 3).mean(axis=0)
        # paint a soft vertical gradient of window light over the figure area
        gx0, gx1 = px0 + int((px1-px0)*0.18), px1 - int((px1-px0)*0.02)
        for yy in range(py0, py1):
            k = (yy - py0) / max(1, (py1 - py0))
            col = light * (1.08 - 0.22 * k)
            a[yy, gx0:gx1] = col
        self.wall = a
        self._built = True
        self.box = (gx0, py0, gx1, py1)

    def render(self, t):
        if not self._built:
            self._build_clean()
        arr = self.wall.copy()
        gx0, py0, gx1, py1 = self.box
        win_w = gx1 - gx0
        win_h = py1 - py0
        # breathing light in the window (+ optional blackout for the payoff)
        pulse = 0.94 + 0.06 * math.sin(t * 1.4)
        amp = pulse
        if self.blackout:
            hold = self.dur * 0.44
            if t > hold:
                k = min((t - hold) / 1.2, 1.0)
                amp = pulse * max(0.0, (1 - k) ** 1.5)

        # figure: shoulder-up, waving arm. Fades in as narration reaches it.
        appear = min(1.0, max(0.0, (t - self.act_from) / 0.7))
        if appear > 0.01:
            hr = win_w * 0.175
            cx = (gx0 + gx1) / 2 - win_w * 0.02
            base_y = py1 - win_h * 0.04
            arm = 30 + 34 * math.sin((t - self.act_from) * 2 * math.pi * 1.35)
            lean = math.sin(t * 1.15) * hr * 0.10
            fig = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
            fi = ImageDraw.Draw(fig)
            fc = (12, 9, 7, int(255 * appear))
            draw_figure(fi, cx, base_y, hr, arm, lean, color=fc)
            arr = arr * (1 - 0)  # no-op keeps dtype
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            img.paste(fig, (0, 0), fig)
            arr = np.asarray(img).astype(np.float32)

        if amp < 0.995:
            # dim the window interior, feathering into the wall
            pad = int(win_w * 0.75)
            ry0, ry1 = max(0, py0 - pad), min(RH, py1 + pad)
            rx0, rx1 = max(0, gx0 - pad), min(RW, gx1 + pad)
            yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
            dd = np.sqrt(((xx - (gx0+gx1)/2) / (win_w * 1.15)) ** 2 +
                         ((yy - (py0+py1)/2) / (win_h * 1.35)) ** 2)
            fall = np.clip(1.0 - dd, 0, 1) ** 1.2
            dim = 1.0 - fall * (1.0 - amp)
            arr[ry0:ry1, rx0:rx1] *= dim[:, :, None]
        add_glow(arr, (gx0 + gx1) / 2, (py0 + py1) / 2, win_w * 2.6,
                 (255, 176, 96), 0.30 * amp)
        # slow push-in
        z = 1.0 + 0.045 * (t / max(self.dur, 0.1))
        cw_, ch_ = RW / z, RH / z
        cx0 = (RW - cw_) / 2
        cy0 = (RH - ch_) / 2
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        im = im.crop((int(cx0), int(cy0), int(cx0 + cw_), int(cy0 + ch_)))
        im = im.resize((RW, RH), Image.LANCZOS)
        return np.asarray(im).astype(np.float32)


MARGIN = 240          # extra canvas width for drift, cropped per frame
CW = RW + MARGIN*2


class StreetScene:
    """Looking across a narrow street at night: one lit window, a figure that waves."""

    def __init__(self, duration, mode="wave"):
        self.dur, self.mode = duration, mode
        self.rng = random.Random(7)
        self.street_y = int(RH * 0.80)
        # --- far building geometry
        self.bx0, self.bx1 = int(CW * 0.10), int(CW * 0.92)
        self.by0, self.by1 = int(RH * 0.06), self.street_y + int(RH * 0.03)
        self.cols, self.rows = 3, 2
        self.win_w = (self.bx1 - self.bx0) * 0.46 / self.cols
        self.win_h = (self.by1 - self.by0) * 0.58 / self.rows
        self.gx = self.bx0 + (self.bx1 - self.bx0) * 0.26
        self.gy = self.by0 + (self.by1 - self.by0) * 0.16
        self.lit = (1, 0)
        self.stars = [(self.rng.uniform(0, CW), self.rng.uniform(0, RH*0.35),
                       self.rng.uniform(0.3, 1.0)) for _ in range(70)]

    def win_rect(self, c, r):
        x = self.gx + c * self.win_w * 1.42
        y = self.gy + r * self.win_h * 1.55
        return x, y, x + self.win_w, y + self.win_h

    def light_level(self, t):
        base = 0.94 + 0.06 * math.sin(t * 1.35)
        if self.mode != "out":
            return base
        hold = self.dur * 0.46
        if t < hold:
            return base
        k = min((t - hold) / 1.15, 1.0)
        return max(0.0, (1 - k) ** 1.6) * base

    def render(self, t):
        # ---- sky
        arr = vgrad(CW, RH, hexc("060a14"), hexc("121a2e"))
        for (sx, sy, sb) in self.stars:
            add_glow(arr, sx, sy, 7, (210, 225, 255), 0.5 * sb)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)

        # ---- far building
        d.rectangle([self.bx0, self.by0, self.bx1, self.by1], fill=hexc("161b27"))
        d.rectangle([self.bx0 - 14, self.by0 - 12, self.bx1 + 14, self.by0], fill=hexc("1b2130"))
        for i in range(11):
            yy = self.by0 + (self.by1 - self.by0) * (i + 0.5) / 11
            d.line([(self.bx0, yy), (self.bx1, yy)], fill=hexc("12161f"), width=2)

        amp = self.light_level(t)
        for c in range(self.cols):
            for r in range(self.rows):
                x0, y0, x1, y1 = self.win_rect(c, r)
                d.rectangle([x0 - 7, y0 - 7, x1 + 7, y1 + 7], fill=hexc("242a37"))
                if (c, r) == self.lit:
                    col = tuple(int(v * amp) for v in (255, 202, 132))
                    d.rectangle([x0, y0, x1, y1], fill=col)
                    # cross mullion
                    d.line([((x0+x1)/2, y0), ((x0+x1)/2, y1)],
                           fill=tuple(int(v * amp * 0.35) + 12 for v in (255, 202, 132)),
                           width=max(2, int((x1-x0)*0.03)))
                else:
                    d.rectangle([x0, y0, x1, y1], fill=hexc("0b0f18"))
                d.line([(x0 - 12, y1 + 9), (x1 + 12, y1 + 9)], fill=hexc("1e2431"), width=5)

        # ---- the figure, inside the lit window
        x0, y0, x1, y1 = self.win_rect(*self.lit)
        hr = (x1 - x0) * 0.155
        wx = (x0 + x1) / 2 + (x1 - x0) * 0.03
        arm = -6 + 30 * math.sin(t * 2 * math.pi * 1.45)
        lean = math.sin(t * 1.2) * hr * 0.10
        if amp > 0.02:
            fig = Image.new("RGBA", img.size, (0, 0, 0, 0))
            fi = ImageDraw.Draw(fig)
            fc = tuple(int(v * amp) for v in (26, 16, 9)) + (255,)
            draw_figure(fi, wx, y1 + 6, hr, arm, lean, color=fc)
            img.paste(fig, (0, 0), fig)
        arr = np.asarray(img).astype(np.float32)

        # ---- light spill
        if amp > 0.02:
            add_glow(arr, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 5.0, (255, 168, 88), 0.50 * amp)
            add_glow(arr, (x0 + x1) / 2, self.street_y + RH * 0.06, (x1 - x0) * 6.5,
                     (255, 156, 78), 0.26 * amp)

        # ---- street
        s = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        ds = ImageDraw.Draw(s)
        ds.rectangle([0, self.street_y, CW, RH], fill=hexc("0d1119"))
        for i in range(16):
            yy = self.street_y + (RH - self.street_y) * (i / 14) ** 1.7
            ds.line([(0, yy), (CW, yy)], fill=hexc("111623"), width=3)
        for i in range(26):
            xx = CW * i / 26
            ds.line([(xx, self.street_y), (xx - 70, RH)], fill=hexc("111623"), width=3)
        arr = np.asarray(s).astype(np.float32)
        # wet-asphalt reflection of the lit window
        if amp > 0.02:
            for k in range(7):
                yy = self.street_y + 18 + k * (RH - self.street_y) / 8.5
                add_glow(arr, (x0 + x1) / 2 + math.sin(k * 2.3) * 6, yy,
                         (x1 - x0) * (0.9 - k * 0.08), (255, 150, 74),
                         (0.16 - k * 0.018) * amp)

        # ---- foreground room frame (nearer = faster drift, drawn later)
        # soft dark edges (vignette-like), no hard boxes
        ew = int(CW * 0.10)
        side = np.clip(np.linspace(1.0, 0.0, ew)[None, :, None] * 1.35, 0, 1) ** 1.6
        arr[:, :ew] *= (1 - side)
        arr[:, CW-ew:] *= (1 - side[:, ::-1])
        top = (np.clip(np.linspace(1.0, 0.0, int(RH*0.16))[:, None, None] * 1.4, 0, 1) ** 1.5)
        arr[:int(RH*0.16)] *= (1 - top)

        # ---- drift + crop (no wrap: slide inside the wide canvas)
        off = int(MARGIN * 0.5 + t * 1.4) % (MARGIN * 2)
        off = min(off, CW - RW)
        return arr[:, off:off + RW]


class CityScene:
    """Skyline where windows go dark one at a time, outward from the middle."""

    def __init__(self, duration):
        self.dur = duration
        self.rng = random.Random(23)
        self.windows = []
        self.buildings = []
        x = -80
        # three depth layers: far (bright, small), mid, near (dark, big)
        for layer, (nwin, wsize, prob, shade) in enumerate(
                [(0, 6, 0.22, 26), (1, 9, 0.34, 18), (2, 13, 0.30, 12)]):
            ybase = RH * (0.80 - layer * 0.012)
            while x < CW + 80:
                w = self.rng.randint(int(CW * 0.035), int(CW * 0.085))
                h = self.rng.randint(int(RH * (0.20 + layer * 0.07)), int(RH * (0.42 + layer * 0.16)))
                y0 = int(ybase) - h
                self.buildings.append((layer, x, y0, x + w, int(ybase),
                                       (shade, shade + 4, shade + 13)))
                if layer > 0:
                    ww = wh = wsize
                    gx, gy = x + ww * 1.6, y0 + wh * 1.8
                    while gx + ww < x + w - ww:
                        yy = gy
                        while yy + wh < ybase - wh:
                            if self.rng.random() < prob:
                                self.windows.append([gx, yy, gx + ww, yy + wh, layer, None])
                            yy += wh * 2.35
                        gx += ww * 2.4
                x += w + self.rng.randint(6, 22)
            x = -80
        self.buildings.sort(key=lambda b: b[0])

        # the window that stays lit: mid layer, near centre
        mid = [w for w in self.windows if w[4] == 1]
        self.mine = min(mid, key=lambda q: abs((q[0] + q[2]) / 2 - CW * 0.5))
        self.mine[1] -= 4
        self.windows = [w for w in self.windows if w is not self.mine]
        # lights go out in a ripple, outward from the lit window
        cx = (self.mine[0] + self.mine[2]) / 2
        for w in self.windows:
            dx = abs((w[0] + w[2]) / 2 - cx) / (CW * 0.55)
            dy = abs((w[1] + w[3]) / 2 - (self.mine[1] + self.mine[3]) / 2) / RH
            d = min(1.0, math.hypot(dx, dy))
            w[5] = 0.6 + d * duration * 0.80 + self.rng.uniform(-0.35, 0.35)

    def render(self, t):
        arr = vgrad(CW, RH, hexc("05080f"), hexc("0e1830"))
        add_glow(arr, CW * 0.5, RH * 0.95, CW * 1.1, (26, 44, 84), 0.55)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for (layer, x0, y0, x1, y1, c) in self.buildings:
            d.rectangle([x0, y0, x1, y1], fill=c)
            d.rectangle([x0, y0, x1, y0 + 4], fill=tuple(min(255, v + 6) for v in c))
        arr = np.asarray(img).astype(np.float32)

        for (x0, y0, x1, y1, layer, off_t) in self.windows:
            if t >= off_t:
                continue
            on = 1.0 if t < off_t - 0.22 else max(0.0, (off_t - t) / 0.22)
            base = (238, 186, 120) if layer == 1 else (198, 158, 104)
            arr[int(y0):int(y1), int(x0):int(x1)] = np.array(base) * (0.85 * on)

        # your window: stays lit, glows
        x0, y0, x1, y1 = self.mine[:4]
        arr[int(y0):int(y1), int(x0):int(x1)] = np.array((255, 214, 150))
        add_glow(arr, (x0 + x1) / 2, (y0 + y1) / 2, 150, (255, 178, 96), 0.95)
        add_glow(arr, (x0 + x1) / 2, (y0 + y1) / 2, 46, (255, 236, 200), 0.55)

        off = min(int(MARGIN * 0.5 + t * 2.0) % (MARGIN * 2), CW - RW)
        return arr[:, off:off + RW]


# --------------------------------------------------------------- timeline
CLIPS = ["a1", "a2", "a3", "a4", "a5", "a6"]
clips = {k: os.path.join(ROOT, "audio", k + ".mp3") for k in CLIPS}


def dur_of(path):
    p = subprocess.run([FF, "-i", path, "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"time=(\d+):(\d+):([\d.]+)", p.stderr)
    h, mi, s = m[-1]
    return int(h) * 3600 + int(mi) * 60 + float(s)


cdur = {k: dur_of(v) for k, v in clips.items()}
start, t = {}, 0.0
for k in CLIPS:
    start[k] = t
    t += cdur[k] + GAP
TOTAL = t - GAP
print("durations:", {k: round(v, 2) for k, v in cdur.items()})
print("narration total: %.2fs" % TOTAL)

# (kind, source, clip, offset, camera move, caption, caption style)
PLAN = [
    ("still", "shot1",  "a1", 0.0,  "in",    "3:47 AM",                              "big"),
    ("still", "shot10", "a1", 3.2,  "out",   "Every window in the city is dark.",    "sub"),
    ("scene", "plaza",  "a1", 5.6,  None,    "Except two. Someone is waving.",       "sub"),
    ("still", "shot2",  "a2", 0.0,  "in",    "What if you never needed to sleep?",   "big"),
    ("still", "shot8",  "a2", 4.2,  "in",    "You switch off the light.",            "sub"),
    ("still", "shot2",  "a2", 7.6,  "out",   "You close your eyes. Nothing.",        "sub"),
    ("still", "shot2",  "a2", 10.6, "in",    "No yawn. No heavy eyelids.",           "sub"),
    ("still", "shot1",  "a2", 12.2, None,    "Not that night. Not ever.",            "big"),
    ("still", "guitar", "a3", 0.0,  "in",    "Eight extra hours. Every night.",      "sub"),
    ("still", "shot9",  "a3", 3.9,  "in",    "2,920 hours a year.",                  "big"),
    ("still", "guitar", "a3", 7.8,  "out",   "Enough class time to learn Spanish",   "sub"),
    ("still", "shot9",  "a3", 11.2, "in",    "four times over.",                     "sub"),
    ("still", "guitar", "a4", 0.0,  "right", "You master the guitar.",               "sub"),
    ("still", "shot9",  "a4", 2.6,  "left",  "You finish every game you own.",       "sub"),
    ("still", "shot3",  "a4", 4.8,  "in",    "But here's the part nobody warns you about.", "sub"),
    ("scene", "city",   "a5", 0.0,  None,    "The world shuts down without you.",    "sub"),
    ("still", "ocean",  "a5", 5.0,  "in",    "One window at a time.",                "big"),
    ("still", "jellyfish", "a5", 7.8, "in",  "Sleep is 450 million years old.",      "sub"),
    ("still", "jellyfish", "a5", 11.4, "out","Even jellyfish do it.",                "big"),
    ("still", "ocean",  "a5", 13.6, "in",    "And they don't have a brain.",         "sub"),
    ("still", "lab1964", "a6", 0.0, "in",    "1964. The most famous attempt.",       "sub"),
    ("still", "lab1964", "a6", 3.6, "out",   "Eleven days awake.",                   "big"),
    ("still", "shot10", "a6", 6.4,  "left",  "You should be the only one.",          "sub"),
    ("scene", "plaza",  "a6", 8.8,  None,    "You wave back.",                       "sub"),
    ("scene", "lightout", "a6", 12.4, None,  "The light goes out.",                  "sub"),
    ("still", "shot6",  "a6", 15.8, "out",   "How long could you last?",             "big"),
]

segments = []
for i, (kind, src, clip, off, move, cap, style) in enumerate(PLAN):
    s = start[clip] + off
    if i + 1 < len(PLAN):
        nk, ns, nc, noff, *_ = PLAN[i + 1]
        e = start[nc] + noff
    else:
        e = TOTAL + 0.8
    segments.append(dict(kind=kind, src=src, t0=s, t1=max(e, s + 0.8),
                         move=move, cap=cap, style=style))

print("cuts: %d over %.1fs (%.1f cuts/min)"
      % (len(segments), TOTAL, len(segments) / TOTAL * 60))

os.makedirs(os.path.join(ROOT, "render"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "frames"), exist_ok=True)


# ------------------------------------------------------------ encode fns
def encode_raw(frames_iter, out_path):
    p = subprocess.Popen(
        [FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "21",
         "-pix_fmt", "yuv420p", "-an", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    for fr in frames_iter:
        p.stdin.write(fr.tobytes())
        n += 1
    p.stdin.close()
    err = p.stderr.read().decode()[-1500:]
    if p.wait() != 0:
        sys.exit("encode failed: " + err)
    return n


def caption_on(img_arr, text, style):
    im = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    size = 96 if style == "big" else 46
    font = ImageFont.truetype(F_BIG if style == "big" else F_SUB, size)
    # wrap
    maxw = W * 0.86
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if d.textlength(test, font=font) <= maxw:
            cur = test
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    lh = size * 1.18
    y = H - 140 - lh * len(lines)
    for ln in lines:
        tw = d.textlength(ln, font=font)
        x = (W - tw) / 2
        # shadow
        for ox, oy in ((0, 0),):
            d.text((x, y), ln, font=font, fill=(0, 0, 0),
                   stroke_width=6, stroke_fill=(0, 0, 0))
        d.text((x, y), ln, font=font, fill=(255, 255, 255))
        y += lh
    return np.asarray(im).astype(np.float32)


def render_still(seg, i):
    d = seg["t1"] - seg["t0"]
    frames = int(round(d * FPS))
    src = still_path(seg["src"])
    step = 0.00045
    mv = seg["move"] or "in"
    if mv == "in":
        z, x, y = f"min(1.02+{step}*on,1.16)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mv == "out":
        z, x, y = f"max(1.16-{step}*on,1.02)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mv == "right":
        z, x, y = "1.14", f"(iw-iw/zoom)*(on/{frames})", "ih/2-(ih/zoom/2)"
    elif mv == "left":
        z, x, y = "1.14", f"(iw-iw/zoom)*(1-on/{frames})", "ih/2-(ih/zoom/2)"
    else:
        z, x, y = "1.06", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    capfile = os.path.join(ROOT, "frames", f"cap{i}.txt")
    open(capfile, "w").write(seg["cap"])
    size = 96 if seg["style"] == "big" else 46
    font = F_BIG if seg["style"] == "big" else F_SUB
    a = (f"if(lt(t,0.45),t/0.45,if(lt(t,{d-0.65:.2f}),1,"
         f"if(lt(t,{d-0.15:.2f}),({d-0.15:.2f}-t)/0.5,0)))")
    vf = (f"scale={W*2}:-2,zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},"
          f"vignette=PI/4.6,noise=alls=3:allf=t+u,"
          f"eq=contrast=1.06:saturation=1.04,"
          f"drawtext=fontfile={font}:textfile={capfile}:fontcolor=white:fontsize={size}"
          f":borderw=3:bordercolor=black@0.55:line_spacing=8"
          f":x=(w-text_w)/2:y=h-text_h-140:alpha='{a}'")
    out = os.path.join(ROOT, "render", f"seg{i:02d}.mp4")
    p = subprocess.run([FF, "-y", "-loop", "1", "-i", src, "-t", f"{d:.3f}",
                        "-vf", vf, "-r", str(FPS), "-c:v", "libx264",
                        "-preset", "medium", "-crf", "21",
                        "-pix_fmt", "yuv420p", "-an", out],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("still failed: " + p.stderr[-1500:])
    return out


def render_scene(seg, i):
    d = seg["t1"] - seg["t0"]
    n = int(round(d * FPS))
    if seg["src"] == "wave":
        sc = StreetScene(d, mode="wave")
    elif seg["src"] == "plaza":
        sc = PlazaScene(d, act_from=0.6)
    elif seg["src"] == "lightout":
        sc = PlazaScene(d, act_from=0.0, blackout=True)
    else:
        sc = CityScene(d)
    rng = np.random.default_rng(5)
    out = os.path.join(ROOT, "render", f"seg{i:02d}.mp4")

    def gen():
        for f in range(n):
            t = f / FPS
            arr = sc.render(t)
            arr = finish(arr, rng)
            # caption fade in/out
            alpha = min(1.0, t / 0.45, max(0.0, (d - t) / 0.5))
            if alpha > 0.01:
                over = caption_on(arr.astype(np.float32), seg["cap"], seg["style"])
                arr = (arr * (1 - alpha) + over * alpha).astype(np.float32)
            yield np.clip(arr, 0, 255).astype(np.uint8)

    frames = encode_raw(gen(), out)
    print(f"    scene '{seg['src']}': {frames} frames rendered")
    return out


parts = []
for i, seg in enumerate(segments):
    if seg["kind"] == "still":
        p = render_still(seg, i)
    else:
        p = render_scene(seg, i)
    parts.append(p)
    print(f"  seg{i:02d} {seg['kind']:5s} {seg['src']:10s} {seg['t1']-seg['t0']:5.2f}s  '{seg['cap']}'")

# ------------------------------------------------------------- audio mix
print("mixing audio...")
narr_in, nf = [], []
for i, k in enumerate(CLIPS):
    narr_in += ["-i", clips[k]]
    nf.append(f"[{i}:a]aresample=48000,apad=pad_dur={GAP}[n{i}]")
chain = "".join(f"[n{i}]" for i in range(len(CLIPS)))
chain += f"concat=n={len(CLIPS)}:v=0:a=1[narr]"

cuts = [s["t0"] for s in segments[1:]]
idx = len(CLIPS)
extra = ["-f", "lavfi", "-i", f"sine=frequency=55:duration={TOTAL+2}"]
i_dr = idx; idx += 1
extra += ["-f", "lavfi", "-i", f"sine=frequency=82.4:duration={TOTAL+2}"]
i_dr2 = idx; idx += 1
extra += ["-f", "lavfi", "-i", f"anoisesrc=color=brown:duration={TOTAL+2}:amplitude=0.5"]
i_nz = idx; idx += 1

filters = nf + [chain]
filters.append(f"[{i_nz}:a]asplit={len(cuts)}" + "".join(f"[nz{j}]" for j in range(len(cuts))))
imp = []
for j, ct in enumerate(cuts):
    filters.append(
        f"[nz{j}]highpass=f=90,lowpass=f=900,afade=t=in:st=0:d=0.02,"
        f"afade=t=out:st=0.05:d=0.35,volume=0.62,"
        f"adelay={int(ct*1000)}|{int(ct*1000)}[imp{j}]")
    imp.append(f"[imp{j}]")
filters.append(f"[{i_dr}:a]volume=0.042,tremolo=f=0.12:d=0.5[dr]")
filters.append(f"[{i_dr2}:a]volume=0.020,tremolo=f=0.11:d=0.6[dr2]")
mix = "[narr]" + "".join(imp) + "[dr][dr2]"
filters.append(f"{mix}amix=inputs={len(cuts)+3}:duration=first:dropout_transition=0,"
               f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=out:st={TOTAL-0.7:.2f}:d=0.7[aout]")
p = subprocess.run([FF, "-y"] + narr_in + extra +
                   ["-filter_complex", ";".join(filters), "-map", "[aout]",
                    "-c:a", "aac", "-b:a", "192k",
                    os.path.join(ROOT, "render", "mix.m4a")],
                   capture_output=True, text=True)
if p.returncode != 0:
    sys.exit("audio failed: " + p.stderr[-2000:])

# ---------------------------------------------------------------- concat
lst = os.path.join(ROOT, "render", "list.txt")
open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in parts))
p = subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", os.path.join(ROOT, "render", "mix.m4a"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart",
                    os.path.join(ROOT, "out.mp4")],
                   capture_output=True, text=True)
if p.returncode != 0:
    sys.exit("concat failed: " + p.stderr[-1500:])

json.dump({"total": TOTAL, "cuts": len(segments),
           "segments": segments},
          open(os.path.join(ROOT, "render", "timeline.json"), "w"), indent=1)
print("\nDONE -> %s  %.1fs  %d cuts" % (os.path.join(ROOT, "out.mp4"), TOTAL, len(segments)))
