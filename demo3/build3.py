#!/usr/bin/env python3
"""
build3.py - fully procedural animation. NO source images, no AI stills.

Every frame is drawn in code: characters are jointed rigs (hips, knees,
shoulders, elbows) that walk, sit, breathe and wave; environments are
built from primitives; a virtual 2D camera pans, dollies, zooms and
handheld-shakes. Only the narration is pre-recorded audio.

  Scenes : CITY · STREET · BEDROOM · ROOM · OCEAN · LAB · FINALE
  Output : 1920x1080 30fps H.264
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
GAP = 0.35
F_BIG = os.path.join(ROOT, "../demo/fonts/Anton.ttf")
F_SUB = os.path.join(ROOT, "../demo/fonts/Inter-SemiBold.ttf")

# ------------------------------------------------------------------ utils
def hexc(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def lerp(a, b, k):
    return a + (b - a) * k


def ease(k):
    return k * k * (3 - 2 * k)


def vgrad(w, h, top, bot):
    t = np.linspace(0, 1, h)[:, None]
    return np.repeat((np.array(top) * (1 - t) + np.array(bot) * t)[:, None, :], w, axis=1)


_GLOW = {}


def glow_sprite(size, color):
    k = (size, tuple(color))
    if k not in _GLOW:
        y, x = np.mgrid[0:size, 0:size]
        c = (size - 1) / 2
        d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
        _GLOW[k] = (np.clip(1 - d, 0, 1) ** 2.2)[:, :, None] * np.array(color)[None, None, :]
    return _GLOW[k]


def add_glow(arr, cx, cy, size, color, strength):
    g = glow_sprite(int(size), color)
    h, w = g.shape[:2]
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, arr.shape[1]), min(y0 + h, arr.shape[0])
    if xb <= xa or yb <= ya:
        return
    arr[ya:yb, xa:xb] += g[ya - y0:ya - y0 + (yb - ya), xa - x0:xa - x0 + (xb - xa)] * strength


def vignette():
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    return np.clip(1 - d ** 4.4 * 0.9, 0.3, 1)[:, :, None]


VIG = vignette()


# ------------------------------------------------------------------ camera
class Cam:
    """2D virtual camera: world point -> screen point."""

    def __init__(self, x=0.0, y=0.0, z=1.0, rot=0.0):
        self.x, self.y, self.z, self.rot = x, y, z, rot

    @staticmethod
    def blend(a, b, k):
        k = ease(k)
        return Cam(lerp(a.x, b.x, k), lerp(a.y, b.y, k),
                   lerp(a.z, b.z, k), lerp(a.rot, b.rot, k))

    def add_shake(self, amp, t):
        return Cam(self.x + math.sin(t * 37.0) * amp,
                   self.y + math.sin(t * 29.0 + 1.3) * amp, self.z, self.rot)


class Pen:
    """Draws in world coordinates through the camera."""

    def __init__(self, img, cam):
        self.img = img
        self.d = ImageDraw.Draw(img, "RGBA")
        self.c = cam

    def p(self, x, y):
        return ((x - self.c.x) * self.c.z + W / 2,
                (y - self.c.y) * self.c.z + H / 2)

    def s(self, v):
        return v * self.c.z

    def line(self, p1, p2, width, fill):
        a, b = self.p(*p1), self.p(*p2)
        self.d.line([a, b], fill=fill, width=max(1, int(self.s(width))), joint="curve")

    def circle(self, c, r, fill):
        x, y = self.p(*c)
        rr = self.s(r)
        self.d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)

    def rect(self, x0, y0, x1, y1, fill):
        a, b = self.p(x0, y0), self.p(x1, y1)
        self.d.rectangle([a, b], fill=fill)

    def poly(self, pts, fill):
        self.d.polygon([self.p(*q) for q in pts], fill=fill)

    def ellipse(self, c, rx, ry, fill, rot=0.0):
        # axis-aligned ellipse in world units
        x, y = self.p(*c)
        rxs, rys = self.s(rx), self.s(ry)
        if rot == 0.0:
            self.d.ellipse([x - rxs, y - rys, x + rxs, y + rys], fill=fill)
        else:
            n = 24
            pts = [(x + rxs * math.cos(a) * math.cos(rot) - rys * math.sin(a) * math.sin(rot),
                    y + rxs * math.cos(a) * math.sin(rot) + rys * math.sin(a) * math.cos(rot))
                   for a in [2 * math.pi * i / n for i in range(n)]]
            self.d.polygon(pts, fill=fill)


pend_arr = np.zeros((H, W, 3), dtype=np.float32)   # active frame buffer for glows


# ------------------------------------------------------------------- rig
# Angle convention: 0 = pointing DOWN, +pi/2 = pointing to the figure's right,
# pi = pointing UP. Limbs are drawn hip->knee->ankle and shoulder->elbow->hand.
def person(pen, x, ground_y, h, pose, color=(8, 10, 16), rim=None):
    head_r = h * 0.082
    spine = h * 0.40
    hip_w = h * 0.075
    lean = pose.get("lean", 0.0)
    sitting = pose.get("sit", False)

    hx = x
    hy = ground_y - (h * 0.30 if sitting else h * 0.47)
    sx = hx + math.sin(lean) * spine
    sy = hy - math.cos(lean) * spine
    hd = (sx + math.sin(lean) * head_r * 2.0, sy - math.cos(lean) * head_r * 2.0)

    def limb(o, a1, l1, a2, l2, w):
        j = (o[0] + math.sin(a1) * l1, o[1] + math.cos(a1) * l1)
        e = (j[0] + math.sin(a2) * l2, j[1] + math.cos(a2) * l2)
        pen.line(o, j, w, color)
        pen.line(j, e, w * 0.9, color)
        pen.circle(j, w * 0.5, color)
        pen.circle(e, w * 0.46, color)
        return e

    # far limbs behind the body
    limb((hx - hip_w * 0.55, hy), pose.get("thigh_far", 0.05), h * 0.24,
         pose.get("shin_far", pose.get("thigh_far", 0.05) + pose.get("knee_far", 0.0)),
         h * 0.235, h * 0.082)
    limb((sx - hip_w * 0.75, sy + h * 0.03), pose.get("arm_far", 0.16), h * 0.19,
         pose.get("fore_far", 0.10), h * 0.175, h * 0.062)

    # torso: tapered, wider shoulders
    pen.poly([(hx - hip_w, hy + 4), (hx + hip_w, hy + 4),
              (sx + hip_w * 1.55, sy), (sx - hip_w * 1.55, sy)], color)
    pen.circle((sx, sy), hip_w * 1.35, color)
    pen.circle((hx, hy), hip_w * 0.95, color)

    # near limbs
    limb((hx + hip_w * 0.55, hy), pose.get("thigh", 0.05), h * 0.24,
         pose.get("shin", pose.get("thigh", 0.05) + pose.get("knee", 0.0)),
         h * 0.235, h * 0.086)
    hand = limb((sx + hip_w * 0.75, sy + h * 0.03), pose.get("arm", 0.14), h * 0.19,
                pose.get("fore", 0.10), h * 0.175, h * 0.066)

    # neck + head
    pen.line((sx, sy - hip_w * 0.4), hd, h * 0.05, color)
    pen.circle(hd, head_r, color)
    if rim:
        pen.ellipse((hd[0] + head_r * 0.3, hd[1] - head_r * 0.12),
                    head_r * 0.5, head_r * 0.6, rim)
    return hd, (sx, sy), hand


def walk_pose(phase):
    s = math.sin(phase)
    return {
        "thigh": s * 0.55, "shin": s * 0.55 - max(0.0, -s) * 0.75,
        "thigh_far": -s * 0.55, "shin_far": -s * 0.55 + max(0.0, s) * 0.75,
        "lean": -0.08,
        "arm": 0.10 - s * 0.55, "fore": 0.35,
        "arm_far": 0.10 + s * 0.55, "fore_far": 0.35,
    }


def wave_pose(t, rate=1.45):
    sw = math.sin(t * 2 * math.pi * rate)
    return {
        "thigh": 0.06, "shin": 0.04, "thigh_far": -0.06, "shin_far": 0.03,
        "lean": 0.0,
        "arm": 2.35 + sw * 0.10,        # raised out to the side
        "fore": 0.30 + sw * 0.55,       # forearm swings back and forth
        "arm_far": 0.14, "fore_far": 0.12,
    }


def stand_pose(t, breathe=True):
    b = math.sin(t * 1.35) * 0.03 if breathe else 0.0
    return {"thigh": 0.05, "shin": 0.03, "thigh_far": -0.05, "shin_far": 0.02,
            "lean": b * 0.4, "arm": 0.13 + b, "fore": 0.10,
            "arm_far": 0.13 - b, "fore_far": 0.10}


def sit_pose(t, typing=True):
    tap = math.sin(t * 2 * math.pi * 4.2) * 0.12 if typing else 0.0
    return {"sit": True,
            "thigh": 1.45, "shin": 0.05, "thigh_far": 1.40, "shin_far": 0.10,
            "lean": 0.16,
            "arm": 1.05 + tap, "fore": 0.85 - tap * 0.5,
            "arm_far": 1.00 - tap, "fore_far": 0.80 + tap * 0.5}


# ----------------------------------------------------------------- scenes
# Coordinate convention: world units == screen pixels at z=1, origin at frame
# centre. So y=-540 is the top edge, y=+540 the bottom. Camera moves are small
# offsets from (0,0) plus a zoom, which keeps every layout predictable.
TOP, BOT = -540, 540


class Scene:
    def __init__(self, dur):
        self.dur = dur

    def draw(self, pen, t):
        raise NotImplementedError


def shadow(pen, x, y, rx, ry=None, a=110):
    pen.ellipse((x, y), rx, ry if ry is not None else rx * 0.16, (0, 0, 0, a))


class CityScene(Scene):
    """Three parallax skyline layers; windows switch off in a ripple."""

    def __init__(self, dur, stay_lit=False, ripple_from=0.55):
        super().__init__(dur)
        self.rng = random.Random(41)
        self.layers, self.windows = [], []
        for li, (base_y, hmin, hmax, shade, prob, wsize) in enumerate([
                (330, 210, 400, 24, 0.15, 7),     # far
                (410, 280, 470, 16, 0.22, 10),    # mid
                (480, 340, 560, 10, 0.20, 13)]):  # near
            builds, x = [], -2200
            while x < 2200:
                bw = self.rng.uniform(80, 190) * (1 + li * 0.28)
                bh = self.rng.uniform(hmin, hmax)
                y0 = base_y - bh
                builds.append((x, y0, x + bw, base_y + 120, shade + li * 4))
                if li > 0:
                    w, hh = wsize, wsize * 1.25
                    gx = x + w * 2.0
                    while gx + w < x + bw - w * 1.4:
                        yy = y0 + hh * 1.9
                        while yy + hh < base_y - hh:
                            if self.rng.random() < prob:
                                self.windows.append([gx, yy, gx + w, yy + hh, li, 0.0])
                            yy += hh * 2.5
                        gx += w * 2.6
                x += bw + self.rng.uniform(10, 34)
            self.layers.append(builds)
        self.stay_lit = stay_lit
        mid = [w for w in self.windows if w[4] == 2]
        self.mine = min(mid, key=lambda q: abs((q[0] + q[2]) / 2)) if mid else None
        if self.mine:
            self.windows = [w for w in self.windows if w is not self.mine]
            cx = (self.mine[0] + self.mine[2]) / 2
            cy = (self.mine[1] + self.mine[3]) / 2
            for w in self.windows:
                d = min(1.0, math.hypot(((w[0] + w[2]) / 2 - cx) / 1500,
                                        ((w[1] + w[3]) / 2 - cy) / 700))
                w[5] = ripple_from + d * dur * 0.55 + self.rng.uniform(-0.3, 0.3)
        self.stars = [(self.rng.uniform(-1400, 1400), self.rng.uniform(-560, -120),
                       self.rng.uniform(0.3, 1.0)) for _ in range(140)]
        self.clouds = [(self.rng.uniform(-1400, 1400), self.rng.uniform(-470, -250),
                        self.rng.uniform(150, 340), self.rng.uniform(0.04, 0.09),
                        self.rng.uniform(3, 9)) for _ in range(6)]

    def draw(self, pen, t):
        hx, hy = pen.p(0, 300)
        add_glow(pend_arr, hx, hy, 1900, (26, 40, 76), 0.55)
        for (sx, sy, sb) in self.stars:
            pen.circle((sx, sy), 1.4 * sb + 0.7, (200, 215, 245, int(160 * sb)))
        mx, my = -520, -400
        add_glow(pend_arr, *pen.p(mx, my), 260, (120, 150, 210), 0.13)
        pen.circle((mx, my), 42, (238, 242, 250, 240))
        pen.circle((mx + 15, my - 11), 36, (16, 24, 42, 255))
        for (cx, cy, cw, ca, spd) in self.clouds:
            drift = ((t * spd + cx + 1600) % 3200) - 1600
            pen.ellipse((drift, cy), cw, cw * 0.13, (140, 160, 200, int(255 * ca)))
            pen.ellipse((drift + cw * 0.4, cy + cw * 0.04), cw * 0.55,
                        cw * 0.10, (150, 170, 205, int(255 * ca * 0.8)))
        for li, builds in enumerate(self.layers):
            for (x0, y0, x1, y1, shade) in builds:
                pen.rect(x0, y0, x1, y1, (shade, shade + 5, shade + 16, 255))
                pen.rect(x0, y0, x1 - (x1 - x0) * 0.75, y1,
                         (shade + 3, shade + 8, shade + 20, 255))
                if li == 2:
                    pen.rect(x0, y0, x1, y0 + 4, (shade + 10, shade + 16, shade + 30, 255))
        for (x0, y0, x1, y1, li, off_t) in self.windows:
            if t >= off_t:
                continue
            on = 1.0 if t < off_t - 0.25 else max(0.0, (off_t - t) / 0.25)
            c = (238, 190, 126) if li == 2 else (190, 156, 106)
            pen.rect(x0, y0, x1, y1, tuple(int(v * on * 0.9) for v in c) + (255,))
        if self.mine:
            x0, y0, x1, y1 = self.mine[:4]
            x0, y0, x1, y1 = x0 - 4, y0 - 5, x1 + 4, y1 + 5
            pen.rect(x0 - 8, y0 - 8, x1 + 8, y1 + 8, (86, 72, 54, 255))
            pen.rect(x0, y0, x1, y1, (255, 222, 164, 255))
            px, py = pen.p((x0 + x1) / 2, (y0 + y1) / 2)
            add_glow(pend_arr, px, py, 300, (255, 170, 88), 0.85)
            add_glow(pend_arr, px, py, 120, (255, 190, 118), 0.75)
            add_glow(pend_arr, px, py, 44, (255, 244, 214), 0.85)
        # ground haze
        pen.rect(-2400, 470, 2400, 760, (12, 17, 30, 255))
        pen.rect(-2400, 470, 2400, 496, (17, 23, 40, 255))


class StreetScene(Scene):
    """Facing facade: one lit window with a waving figure. Push-in on camera."""

    def __init__(self, dur, blackout=False):
        super().__init__(dur)
        self.blackout = blackout
        self.rng = random.Random(9)
        self.bricks = [(self.rng.uniform(-1300, 1300), self.rng.uniform(-560, 120),
                        self.rng.uniform(0.4, 1.0)) for _ in range(240)]
        self.win = (-150, -330, 130, -70)      # lit window world rect

    def draw(self, pen, t):
        amp = 1.0
        if self.blackout:
            hold = self.dur * 0.42
            amp = 1.0 if t < hold else max(0.0, (1 - min((t - hold) / 1.25, 1.0)) ** 1.5)

        # facade
        pen.rect(-1400, -760, 1400, 320, (30, 34, 46, 255))
        for (bx, by, bs) in self.bricks:
            pen.rect(bx, by, bx + 40 * bs, by + 13, (33, 37, 50, 255))
        wx0, wy0, wx1, wy1 = self.win
        px, py = pen.p((wx0 + wx1) / 2, (wy0 + wy1) / 2)
        add_glow(pend_arr, px, py, 1150, (255, 172, 96), 0.32 * amp)

        # dark windows with sills
        for cx in (-900, -560, 340, 680, 1020):
            for cy in (-520, -330, -140):
                pen.rect(cx - 10, cy - 10, cx + 130, cy + 170, (52, 58, 74, 255))
                pen.rect(cx, cy, cx + 120, cy + 160, (11, 14, 22, 255))
                pen.rect(cx - 18, cy + 162, cx + 138, cy + 176, (60, 66, 84, 255))

        # lit window
        pen.rect(wx0 - 12, wy0 - 12, wx1 + 12, wy1 + 12, (58, 64, 82, 255))
        lit = tuple(int(v * amp) for v in (255, 210, 152)) + (255,)
        pen.rect(wx0, wy0, wx1, wy1, lit)
        pen.rect((wx0 + wx1) / 2 - 3, wy0, (wx0 + wx1) / 2 + 3, wy1,
                 tuple(int(v * amp * 0.5) for v in (246, 196, 138)) + (255,))
        if amp > 0.03:
            person(pen, (wx0 + wx1) / 2 - 10, wy1 + 14, (wy1 - wy0) * 0.92,
                   wave_pose(t * 0.9), (18, 12, 8, int(255 * amp)))
        pen.rect(wx0 - 22, wy1 + 12, wx1 + 22, wy1 + 28, (64, 70, 88, 255))

        # street
        pen.rect(-1400, 320, 1400, 700, (15, 19, 28, 255))
        for i in range(13):
            yy = 330 + i * i * 3.4
            pen.rect(-1400, yy, 1400, yy + 4, (19, 24, 36, 255))
        for i in range(-12, 13):
            pen.line((i * 120, 320), (i * 120 - 80, 700), 4, (19, 24, 36, 255))
        # light pool + reflection on the pavement
        if amp > 0.03:
            add_glow(pend_arr, px, pen.p(0, 400)[1], 620, (255, 154, 78), 0.30 * amp)
            for k in range(8):
                _, ry = pen.p(0, 360 + k * 44)
                add_glow(pend_arr, px + math.sin(k * 1.9 + t * 0.8) * 8, ry,
                         150 - k * 12, (255, 150, 74), (0.14 - k * 0.014) * amp)

        # foreground room frame
        pen.rect(-1400, -760, -1130, 700, (3, 5, 9, 255))
        pen.rect(1130, -760, 1400, 700, (3, 5, 9, 255))
        pen.rect(-1400, -760, 1400, -640, (3, 5, 9, 255))
        pen.rect(-1400, 470, 1400, 700, (3, 5, 9, 255))


class BedroomScene(Scene):
    """Lying awake: breathing blanket, blinking eyes, ticking clock, moonbeam."""

    def __init__(self, dur):
        super().__init__(dur)
        self.carpet = [(i * 137 % 1400 - 700, (i * 89) % 400 - 30) for i in range(90)]

    def draw(self, pen, t):
        pen.rect(-1200, -900, 1200, 700, (17, 20, 30, 255))
        pen.rect(-1200, 250, 1200, 700, (13, 15, 23, 255))     # carpet
        for (cx, cy) in self.carpet:
            pen.ellipse((cx, 300 + (cy % 220)), 3, 1.6, (20, 23, 34, 200))
        # window + moonbeam
        pen.rect(-1080, -560, -560, -140, (46, 54, 78, 255))
        pen.rect(-1066, -546, -574, -154, (22, 31, 54, 255))
        pen.circle((-880, -400), 30, (228, 234, 248, 240))
        pen.circle((-866, -412), 25, (22, 31, 54, 255))
        add_glow(pend_arr, *pen.p(-740, -350), 620, (80, 112, 180), 0.20)
        pen.poly([(-1070, 660), (-520, 660), (-330, -120), (-800, -120)], (48, 66, 112, 34))
        # headboard + bed frame
        pen.rect(-330, 60, 690, 200, (40, 46, 66, 255))
        pen.rect(-360, 180, 720, 430, (32, 36, 52, 255))
        pen.rect(-330, 430, 700, 470, (26, 30, 44, 255))
        pen.rect(-330, 470, -300, 540, (26, 30, 44, 255))
        pen.rect(670, 470, 700, 540, (26, 30, 44, 255))
        shadow(pen, 190, 548, 420, 26, 120)
        # pillow
        pen.ellipse((-215, 172), 96, 44, (66, 72, 98, 255))
        # blanket, breathing
        br = math.sin(t * 1.25) * 5
        pen.poly([(-300, 226), (706, 226), (724, 424), (-300, 434)], (36, 42, 64, 255))
        pen.poly([(-300, 226 + br), (706, 226 - br), (722, 410), (-300, 420)], (50, 60, 92, 255))
        pen.poly([(-300, 226 + br), (706, 226 - br), (706, 258), (-300, 264)], (68, 80, 116, 255))
        pen.poly([(-300, 300), (706, 296), (712, 320), (-300, 324)], (44, 53, 82, 255))
        # head + eyes
        hx, hy = -170, 172
        pen.ellipse((hx, hy), 52, 58, (17, 19, 28, 255))
        pen.ellipse((hx + 4, hy + 8), 42, 36, (23, 26, 38, 255))
        blink = (t % 4.2) > 0.2
        if blink:
            for dx in (14, 36):
                pen.ellipse((hx + dx, hy - 4), 9, 7.5, (232, 238, 250, 255))
                pen.circle((hx + dx + 1.5, hy - 4), 3.2, (10, 12, 18, 255))
        else:
            for dx in (8, 30):
                pen.ellipse((hx + dx, hy - 4), 11, 2.6, (74, 80, 100, 255))
        pen.ellipse((hx - 6, hy - 26), 26, 10, (26, 29, 42, 255))
        # bedside table + 7-seg clock
        pen.rect(740, 300, 950, 540, (34, 38, 52, 255))
        pen.rect(770, 240, 920, 312, (16, 18, 26, 255))
        seg = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
               "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abfgcd"}
        cx0, cy0, dw, dh = 788, 252, 20, 46
        for i, ch in enumerate("03:47"):
            ox = cx0 + i * (dw + 9)
            if ch == ":":
                pen.rect(ox, cy0 + dh * 0.22, ox + 6, cy0 + dh * 0.34, (255, 96, 96, 255))
                pen.rect(ox, cy0 + dh * 0.64, ox + 6, cy0 + dh * 0.76, (255, 96, 96, 255))
                continue
            for s in seg[ch]:
                if s == "a": pen.rect(ox, cy0, ox + dw, cy0 + 4, (255, 86, 86, 255))
                if s == "b": pen.rect(ox + dw - 4, cy0, ox + dw, cy0 + dh / 2, (255, 86, 86, 255))
                if s == "c": pen.rect(ox + dw - 4, cy0 + dh / 2, ox + dw, cy0 + dh, (255, 86, 86, 255))
                if s == "d": pen.rect(ox, cy0 + dh - 4, ox + dw, cy0 + dh, (255, 86, 86, 255))
                if s == "e": pen.rect(ox, cy0 + dh / 2, ox + 4, cy0 + dh, (255, 86, 86, 255))
                if s == "f": pen.rect(ox, cy0, ox + 4, cy0 + dh / 2, (255, 86, 86, 255))
                if s == "g": pen.rect(ox, cy0 + dh / 2 - 2, ox + dw, cy0 + dh / 2 + 2, (255, 86, 86, 255))
        add_glow(pend_arr, *pen.p(845, 276), 240, (255, 70, 70), 0.28)
        add_glow(pend_arr, *pen.p(845, 276), 66, (255, 125, 115), 0.34)


class RoomScene(Scene):
    """Walking figure with a real gait, or sitting and typing under a lamp."""

    def __init__(self, dur, mode="walk"):
        super().__init__(dur)
        self.mode = mode

    def draw(self, pen, t):
        pen.rect(-1400, -900, 1400, 700, (14, 17, 26, 255))
        pen.rect(-1400, 300, 1400, 700, (20, 24, 36, 255))
        pen.rect(-1400, 288, 1400, 300, (34, 38, 54, 255))
        pen.ellipse((-60, 470), 520, 60, (17, 20, 31, 255))
        for i in range(-14, 15):
            pen.line((i * 155, 300), (i * 155 - 40, 700), 2, (24, 28, 41, 255))
        pen.rect(-1200, -600, -600, -60, (40, 48, 70, 255))
        pen.rect(-1186, -586, -614, -74, (20, 30, 52, 255))
        pen.circle((-880, -400), 36, (224, 231, 247, 240))
        pen.circle((-864, -414), 29, (20, 30, 52, 255))
        pen.rect(-1210, -70, -590, -50, (52, 58, 80, 255))

        if self.mode == "walk":
            # lamp on a sideboard
            pen.rect(560, 120, 900, 330, (30, 34, 48, 255))
            pen.rect(700, -110, 760, 120, (38, 42, 58, 255))
            pen.poly([(640, -110), (820, -110), (852, -40), (608, -40)], (255, 216, 164, 210))
            add_glow(pend_arr, *pen.p(730, -60), 850, (255, 188, 112), 0.30)
            k = t / self.dur
            x = -620 + 1240 * k
            shadow(pen, x, 322, 118)
            person(pen, x, 320, 430, walk_pose(t * 2 * math.pi * 0.9))
            for i in range(20):
                mx = -300 + (i * 151 + t * 22) % 800
                my = -420 + (math.sin(t * 0.7 + i) * 0.5 + 0.5) * 620
                pen.circle((mx, my), 2.2, (255, 228, 192, 80))
        else:
            # desk + monitor + chair + sitting figure typing
            pen.rect(-320, 130, 620, 180, (46, 40, 36, 255))
            pen.rect(-300, 180, -260, 400, (38, 34, 32, 255))
            pen.rect(560, 180, 600, 400, (38, 34, 32, 255))
            pen.rect(-170, -110, 230, 120, (28, 32, 46, 255))
            pen.rect(-156, -96, 216, 106, (54, 96, 132, 255))
            pen.rect(-130, -80, 60, 90, (40, 78, 112, 255))
            add_glow(pend_arr, *pen.p(30, 5), 700, (74, 156, 200), 0.34)
            pen.rect(-6, 120, 46, 150, (32, 36, 48, 255))
            pen.rect(-90, 150, -40, 150 + 150, (36, 34, 40, 255))     # chair back
            pen.rect(-100, 290, 60, 306, (40, 38, 45, 255))            # seat
            shadow(pen, 40, 402, 110)
            person(pen, -40, 330, 400, sit_pose(t))
            pen.rect(-620, 40, -570, 380, (46, 42, 38, 255))
            pen.poly([(-700, 40), (-490, 40), (-516, -70), (-674, -70)], (255, 216, 164, 200))
            add_glow(pend_arr, *pen.p(-596, 0), 950, (255, 190, 120), 0.30)
            for i in range(18):
                mx = -340 + (i * 197 + t * 16) % 860
                my = -380 + (math.sin(t * 0.6 + i * 1.7) * 0.5 + 0.5) * 600
                pen.circle((mx, my), 2.0, (255, 232, 204, 75))


class OceanScene(Scene):
    """Jellyfish pulsing with undulating tentacles; god rays; drifting motes."""

    def __init__(self, dur, jellyfish=True):
        super().__init__(dur)
        self.jf = jellyfish
        self.rng = random.Random(3)
        self.motes = [(self.rng.uniform(-1200, 1200), self.rng.uniform(-540, 540),
                       self.rng.uniform(0.3, 1.0), self.rng.uniform(4, 22)) for _ in range(130)]
        self.bub = [(self.rng.uniform(-1100, 1100), self.rng.uniform(-540, 540),
                     self.rng.uniform(2, 6), self.rng.uniform(10, 40)) for _ in range(40)]

    def draw(self, pen, t):
        for i in range(6):
            x = -900 + i * 360 + math.sin(t * 0.4 + i) * 46
            wdt = 80 + i * 16
            pen.poly([(x - wdt, -700), (x + wdt, -700),
                      (x + wdt * 2.4, 700), (x + 30, 700)], (120, 180, 230, 11))
        for (mx, my, mb, sp) in self.motes:
            yy = ((my + 540 + t * sp) % 1080) - 540
            pen.circle((mx + math.sin(t * 0.5 + my * 0.01) * 8, yy), 1.8 * mb + 0.6,
                       (198, 228, 255, int(50 + 80 * mb)))
        for (bx, by, br_, sp) in self.bub:
            yy = ((by + 540 - t * sp) % 1080) - 540
            pen.circle((bx + math.sin(t + by) * 4, yy), br_, (190, 235, 250, 55))
        if self.jf:
            for k, (jx, js, jd) in enumerate([(-260, 1.0, 0.0), (300, 0.66, 1.5),
                                              (620, 0.44, 2.8)]):
                pulse = math.sin(t * 2 * math.pi * 0.40 + jd)
                by = -60 + math.sin(t * 0.5 + jd) * 54 + k * 70
                bw = 170 * js * (1 + 0.07 * pulse)
                bh = 108 * js * (1 + 0.11 * pulse)
                add_glow(pend_arr, *pen.p(jx, by), 560 * js, (86, 220, 235), 0.24)
                pen.ellipse((jx, by), bw, bh, (150, 236, 242, 62))
                pen.ellipse((jx, by - bh * 0.12), bw * 0.86, bh * 0.72, (196, 250, 252, 46))
                pen.ellipse((jx, by + bh * 0.22), bw * 0.94, bh * 0.26, (128, 224, 238, 90))
                pen.ellipse((jx - bw * 0.34, by - bh * 0.2), bw * 0.22, bh * 0.3,
                            (225, 250, 253, 60))
                for i in range(9):
                    ox = jx - bw * 0.72 + i * (bw * 1.44 / 8)
                    pts = []
                    for seg in range(8):
                        yy = by + bh * 0.52 + seg * (40 * js)
                        xx = ox + math.sin(t * 1.7 + seg * 0.65 + i * 0.9 + jd) * (8 + seg * 3.2) * js
                        pts.append((xx, yy))
                    for a, b in zip(pts, pts[1:]):
                        pen.line(a, b, 3.0 * js, (168, 240, 246, 135))
                for i in range(4):
                    ox = jx - bw * 0.2 + i * (bw * 0.4 / 3)
                    pts = [(-200 * js, 0)]
                    pts = []
                    for seg in range(6):
                        yy = by + bh * 0.5 + seg * (54 * js)
                        xx = ox + math.sin(t * 1.2 + seg * 0.5 + i + jd) * (10 + seg * 2) * js
                        pts.append((xx, yy))
                    for a, b in zip(pts, pts[1:]):
                        pen.line(a, b, 5.0 * js, (150, 230, 240, 80))


class LabScene(Scene):
    """1964 sleep lab: subject, straps, dials, scrolling waveform, observers."""

    def __init__(self, dur):
        super().__init__(dur)
        self.rng = random.Random(77)

    def draw(self, pen, t):
        pen.rect(-1500, -900, 1500, 700, (18, 20, 29, 255))
        for i in range(-16, 17):
            pen.line((i * 110, -900), (i * 110, 200), 2, (23, 25, 36, 255))
        for j in range(-9, 2):
            pen.line((-1500, j * 110), (1500, j * 110), 2, (23, 25, 36, 255))
        pen.rect(-1500, 200, 1500, 700, (14, 16, 23, 255))

        # machines with spinning dials
        for mx in (-1120, -900):
            pen.rect(mx, -420, mx + 190, 10, (36, 40, 52, 255))
            pen.rect(mx + 14, -404, mx + 176, -270, (11, 15, 21, 255))
            for k in range(3):
                cx = mx + 48 + k * 52
                pen.circle((cx, -338), 17, (62, 68, 84, 255))
                pen.circle((cx, -338), 12, (26, 30, 40, 255))
                ang = t * (1.4 + k * 0.6) + k
                pen.line((cx, -338), (cx + math.cos(ang) * 10, -338 + math.sin(ang) * 10),
                         2.5, (205, 214, 232, 255))
            for k in range(4):
                pen.circle((mx + 40 + k * 38, -240), 6, (90, 70, 60, 255)
                           if (int(t * 2) + k) % 2 else (40, 34, 30, 255))

        # oscilloscope
        pen.rect(-1500, -420, -1150, -180, (32, 36, 48, 255))
        pen.rect(-1484, -404, -1166, -196, (9, 26, 17, 255))
        pts = [(-1484 + i * 5.6,
                -300 + math.sin(i * 0.5 + t * 5.6) * 20 * (0.6 + 0.4 * math.sin(i * 0.18 + t)))
               for i in range(60)]
        for a, b in zip(pts, pts[1:]):
            pen.line(a, b, 3, (92, 244, 152, 255))
        add_glow(pend_arr, *pen.p(-1325, -300), 360, (60, 220, 130), 0.26)

        # bed + subject
        # gurney frame + legs, mattress, sheet
        pen.rect(-460, 300, 700, 330, (58, 62, 76, 255))
        for lx in (-430, 620):
            pen.rect(lx, 330, lx + 22, 470, (46, 50, 62, 255))
            pen.ellipse((lx + 40, 474), 26, 8, (30, 34, 44, 255))
        pen.rect(-450, 250, 700, 306, (92, 96, 112, 255))
        pen.poly([(-450, 250), (700, 250), (700, 306), (-450, 306)], (98, 102, 118, 255))
        pen.poly([(-230, 236), (700, 236), (700, 300), (-230, 306)], (132, 138, 156, 255))
        pen.ellipse((-300, 262), 88, 34, (150, 156, 174, 255))
        hx, hy = -280, 250
        pen.ellipse((hx, hy), 46, 50, (58, 62, 76, 255))
        if (t % 3.4) > 0.2:
            pen.circle((hx + 16, hy - 4), 8, (238, 242, 252, 255))
            pen.circle((hx + 18, hy - 4), 3.0, (10, 12, 18, 255))
        else:
            pen.ellipse((hx + 16, hy - 4), 13, 3, (86, 92, 112, 255))
        for sx in (-190, 210):
            pen.rect(sx, 236, sx + 24, 306, (60, 52, 44, 255))
        # clip wires
        for k in range(3):
            pen.line((hx + 20, hy - 20 + k * 8), (-1000, -140 + k * 20), 2.4, (90, 96, 112, 255))

        # observers
        person(pen, 560, 470, 430, {"thigh": 0.05, "knee": -0.02,
                                    "thigh_far": -0.05, "knee_far": -0.02, "lean": -0.06,
                                    "shoulder": -1.05 + math.sin(t * 1.1) * 0.12,
                                    "elbow": -0.30,
                                    "shoulder_far": 0.08, "elbow_far": 0.12}, (13, 15, 22, 255))
        person(pen, 700, 470, 430, stand_pose(t * 0.9), (13, 15, 22, 255))
        shadow(pen, 630, 474, 230)
        # hanging lamp
        pen.rect(-30, -900, 4, -690, (32, 36, 46, 255))
        pen.poly([(-110, -690), (84, -690), (108, -626), (-134, -626)], (255, 222, 174, 215))
        add_glow(pend_arr, *pen.p(-16, -650), 1400, (255, 202, 142), 0.28)
        pen.ellipse((-20, 420), 240, 70, (255, 198, 140, 16))


# --------------------------------------------------------------- timeline
CLIPS = ["a1", "a2", "a3", "a4", "a5", "a6"]
clips = {k: os.path.join(ROOT, "audio", k + ".mp3") for k in CLIPS}


def dur_of(p):
    r = subprocess.run([FF, "-i", p, "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"time=(\d+):(\d+):([\d.]+)", r.stderr)[-1]
    return int(m[0]) * 3600 + int(m[1]) * 60 + float(m[2])


cdur = {k: dur_of(v) for k, v in clips.items()}
start, _t = {}, 0.0
for k in CLIPS:
    start[k] = _t
    _t += cdur[k] + GAP
TOTAL = _t - GAP
print("narration: %.2fs" % TOTAL)

# shot: (scene, scene_args, clip, offset, caption, style, cam_from, cam_to)
Z = Cam
PLAN = [
    # city opening: slow push toward the skyline
    ("city",   {"stay_lit": False}, "a1", 0.0, "3:47 AM", "big",
     Z(-40, 150, 0.98), Z(-20, 190, 1.10)),
    ("street", {}, "a1", 3.1, "Every window in the city is dark.", "sub",
     Z(-120, -180, 0.86), Z(-20, -220, 1.02)),
    ("street", {}, "a1", 6.0, "Except two. Someone is waving.", "sub",
     Z(-10, -230, 1.16), Z(20, -240, 1.34)),
    ("bedroom", {}, "a2", 0.4, "You close your eyes. Nothing.", "sub",
     Z(150, 150, 0.92), Z(190, 130, 1.00)),
    ("bedroom", {}, "a2", 4.0, "No yawn. No heavy eyelids.", "sub",
     Z(230, 160, 1.00), Z(180, 140, 1.08)),
    ("city",   {"stay_lit": True}, "a2", 7.6, "Not that night. Not ever.", "big",
     Z(10, 160, 1.0), Z(0, 200, 1.14)),
    ("room",   {"mode": "walk"}, "a3", 0.0, "Eight extra hours. Every night.", "sub",
     Z(-120, 20, 0.86), Z(60, 10, 0.94)),
    ("room",   {"mode": "sit"}, "a3", 4.2, "2,920 hours a year.", "big",
     Z(60, 90, 0.90), Z(120, 110, 1.00)),
    ("room",   {"mode": "walk"}, "a3", 8.4, "Enough class time to learn Spanish", "sub",
     Z(160, 10, 0.96), Z(-60, 20, 0.88)),
    ("room",   {"mode": "sit"}, "a4", 0.0, "You master the guitar.", "sub",
     Z(140, 100, 0.98), Z(80, 80, 0.90)),
    ("bedroom", {}, "a4", 2.8, "But here's the part nobody warns you about.", "sub",
     Z(200, 150, 0.98), Z(160, 130, 0.92)),
    ("city",   {"stay_lit": False, "ripple_from": 0.35}, "a5", 0.0,
     "The world shuts down without you.", "sub",
     Z(0, 170, 1.02), Z(-30, 210, 0.90)),
    ("city",   {"stay_lit": False, "ripple_from": 0.2}, "a5", 4.4,
     "One window at a time.", "big", Z(80, 180, 1.10), Z(-60, 210, 0.98)),
    ("ocean",  {"jellyfish": False}, "a5", 7.4, "Sleep is 450 million years old.", "sub",
     Z(0, -60, 0.92), Z(0, -20, 1.04)),
    ("ocean",  {"jellyfish": True}, "a5", 10.8, "Even jellyfish do it.", "big",
     Z(-90, -40, 0.96), Z(40, -70, 1.12)),
    ("ocean",  {"jellyfish": True}, "a5", 13.2, "And they don't have a brain.", "sub",
     Z(140, -70, 1.06), Z(30, -40, 0.94)),
    ("lab",    {}, "a6", 0.0, "1964. The most famous attempt.", "sub",
     Z(0, 200, 0.86), Z(60, 190, 0.94)),
    ("lab",    {}, "a6", 3.4, "Eleven days awake.", "big",
     Z(60, 210, 0.96), Z(-40, 195, 0.88)),
    ("street", {}, "a6", 6.2, "You should be the only one.", "sub",
     Z(30, -240, 1.24), Z(-10, -235, 1.12)),
    ("street", {}, "a6", 9.0, "You wave back.", "sub",
     Z(20, -240, 1.32), Z(25, -238, 1.24)),
    ("street", {"blackout": True}, "a6", 12.2, "And the light goes out.", "sub",
     Z(25, -238, 1.26), Z(40, -245, 1.12)),
    ("city",   {"stay_lit": True}, "a6", 15.4, "How long could you last?", "big",
     Z(0, 170, 1.04), Z(0, 200, 1.18)),
]

segments = []
for i, (scene, args, clip, off, cap, style, c0, c1) in enumerate(PLAN):
    s = start[clip] + off
    if i + 1 < len(PLAN):
        _, _, nc, noff = PLAN[i + 1][:4]
        e = start[nc] + noff
    else:
        e = TOTAL + 0.8
    segments.append(dict(scene=scene, args=args, t0=s, t1=max(e, s + 0.9),
                         cap=cap, style=style, c0=c0, c1=c1))
print("shots: %d over %.1fs (%.1f shots/min)" % (len(segments), TOTAL, len(segments) / TOTAL * 60))

# ---------------------------------------------------------------- render
pend_arr = None   # set per frame (numpy float array being drawn on)


def base_layer(scene, t):
    if scene == "ocean":
        k = 0.5 + 0.5 * math.sin(t * 0.25)
        return vgrad(W, H, (6, 20, 40), (2, 6, 16))
    if scene == "city":
        return vgrad(W, H, (7, 11, 22), (20, 30, 56))
    if scene == "bedroom":
        return vgrad(W, H, (12, 15, 24), (22, 26, 40))
    if scene == "lab":
        return vgrad(W, H, (18, 20, 28), (26, 28, 38))
    return vgrad(W, H, (10, 13, 20), (20, 24, 36))


def caption_layer(frame, text, style, alpha):
    im = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    size = 96 if style == "big" else 46
    font = ImageFont.truetype(F_BIG if style == "big" else F_SUB, size)
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
        d.text(((W - tw) / 2, y), ln, font=font, fill=(0, 0, 0),
               stroke_width=6, stroke_fill=(0, 0, 0))
        d.text(((W - tw) / 2, y), ln, font=font, fill=(255, 255, 255))
        y += lh
    out = np.asarray(im).astype(np.float32)
    return frame * (1 - alpha) + out * alpha


def render_frames(seg, n):
    scene = seg["scene"]
    dur = seg["t1"] - seg["t0"]
    cls = {"city": CityScene, "street": StreetScene, "bedroom": BedroomScene,
           "room": RoomScene, "ocean": OceanScene, "lab": LabScene}[scene]
    inst = cls(dur, **seg["args"])
    rng = np.random.default_rng(9)
    for f in range(n):
        t = seg["t0"] + f / FPS
        lt = f / FPS                      # local time for animation
        k = f / max(1, n - 1)
        cam = Cam.blend(seg["c0"], seg["c1"], k)
        cam = cam.add_shake(0.9 if scene != "city" else 0.5, lt)
        img = Image.fromarray(base_layer(scene, lt).astype(np.uint8))
        global pend_arr
        pend_arr = np.asarray(img).astype(np.float32)
        # draws go to a PIL layer for crisp shapes, then composite over base
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pen = Pen(layer, cam)
        inst.draw(pen, lt)
        lay = np.asarray(layer).astype(np.float32)
        a = lay[:, :, 3:4] / 255.0
        arr = pend_arr * (1 - a) + lay[:, :, :3] * a
        # vignette + grain
        arr = arr * VIG
        arr = arr + rng.normal(0, 3.2, arr.shape)
        arr = np.clip(arr, 0, 255)
        # caption fade
        alpha = min(1.0, (f / FPS) / 0.45, max(0.0, (dur - f / FPS) / 0.5))
        if alpha > 0.01:
            arr = caption_layer(arr.astype(np.float32), seg["cap"], seg["style"], alpha)
        yield np.clip(arr, 0, 255).astype(np.uint8)


def encode(frames, out):
    p = subprocess.Popen([FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                          "-pix_fmt", "yuv420p", "-an", out],
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    err = p.stderr.read().decode()[-1200:]
    if p.wait() != 0:
        sys.exit("encode failed: " + err)


os.makedirs(os.path.join(ROOT, "render"), exist_ok=True)
ONLY = {s.strip() for s in os.environ.get("ONLY_SCENES", "").split(",") if s.strip()}
parts = []
for i, seg in enumerate(segments):
    n = int(round((seg["t1"] - seg["t0"]) * FPS))
    out = os.path.join(ROOT, "render", f"s{i:02d}.mp4")
    if ONLY and seg["scene"] not in ONLY and os.path.exists(out):
        parts.append(out)
        print("  %02d %-8s reused" % (i, seg["scene"]))
        continue
    encode(render_frames(seg, n), out)
    parts.append(out)
    print("  %02d %-8s %5.2fs  %s" % (i, seg["scene"], seg["t1"] - seg["t0"], seg["cap"]))

# ------------------------------------------------------------- audio mix
print("mixing...")
nin, nf = [], []
for i, k in enumerate(CLIPS):
    nin += ["-i", clips[k]]
    nf.append(f"[{i}:a]aresample=48000,apad=pad_dur={GAP}[n{i}]")
chain = "".join(f"[n{i}]" for i in range(len(CLIPS))) + f"concat=n={len(CLIPS)}:v=0:a=1[narr]"
cuts = [s["t0"] for s in segments[1:]]
idx = len(CLIPS)
extra = ["-f", "lavfi", "-i", f"sine=frequency=55:duration={TOTAL+2}"]; i1 = idx; idx += 1
extra += ["-f", "lavfi", "-i", f"sine=frequency=82.4:duration={TOTAL+2}"]; i2 = idx; idx += 1
extra += ["-f", "lavfi", "-i", f"anoisesrc=color=brown:duration={TOTAL+2}:amplitude=0.5"]; i3 = idx
f = nf + [chain, f"[{i3}:a]asplit={len(cuts)}" + "".join(f"[nz{j}]" for j in range(len(cuts)))]
imp = []
for j, ct in enumerate(cuts):
    f.append(f"[nz{j}]highpass=f=90,lowpass=f=900,afade=t=in:st=0:d=0.02,"
             f"afade=t=out:st=0.05:d=0.35,volume=0.6,"
             f"adelay={int(ct*1000)}|{int(ct*1000)}[imp{j}]")
    imp.append(f"[imp{j}]")
f.append(f"[{i1}:a]volume=0.042,tremolo=f=0.12:d=0.5[dr]")
f.append(f"[{i2}:a]volume=0.020,tremolo=f=0.11:d=0.6[dr2]")
mix = "[narr]" + "".join(imp) + "[dr][dr2]"
f.append(f"{mix}amix=inputs={len(cuts)+3}:duration=first:dropout_transition=0,"
         f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=out:st={TOTAL-0.7:.2f}:d=0.7[aout]")
p = subprocess.run([FF, "-y"] + nin + extra + ["-filter_complex", ";".join(f),
                    "-map", "[aout]", "-c:a", "aac", "-b:a", "192k",
                    os.path.join(ROOT, "render", "mix.m4a")], capture_output=True, text=True)
if p.returncode != 0:
    sys.exit("audio failed: " + p.stderr[-1500:])

lst = os.path.join(ROOT, "render", "list.txt")
open(lst, "w").write("".join(f"file '{os.path.abspath(x)}'\n" for x in parts))
p = subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", os.path.join(ROOT, "render", "mix.m4a"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", os.path.join(ROOT, "out.mp4")],
                   capture_output=True, text=True)
if p.returncode != 0:
    sys.exit("concat failed: " + p.stderr[-1200:])
json.dump({"total": TOTAL, "shots": len(segments), "segments": [
    {k: v for k, v in s.items() if k not in ("c0", "c1")} for s in segments]},
    open(os.path.join(ROOT, "render", "timeline.json"), "w"), indent=1)
print("\nDONE -> out.mp4  %.1fs  %d shots  (fully procedural)" % (TOTAL, len(segments)))
