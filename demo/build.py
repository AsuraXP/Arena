#!/usr/bin/env python3
"""
build.py - "What-If" style 2D animation renderer.

Pipeline (mirrors the 8-stage skill in the tutorial, but with $0/local pieces):
  1. script      -> shots.yaml / SHOTS below
  2. voice       -> audio/s*.mp3 (Arena TTS, no ElevenLabs)
  3. shot plan   -> cut list, ~25-30 cuts/min
  4. visuals     -> shots/*.png (AI image gen)
  5. animation   -> ffmpeg zoompan: push-in / pull-out / pan, vignette, grain
  6. sound       -> synth drone + noise impacts on every cut (no sample packs)
  7. mix         -> narration ducking + loudnorm
  8. render      -> 1920x1080 30fps h264 mp4
"""
import json, os, re, subprocess, sys
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.abspath(__file__))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
FONT_BOLD = os.path.join(ROOT, "fonts/Anton.ttf")
FONT_SUB = os.path.join(ROOT, "fonts/Inter-SemiBold.ttf")
GAP = 0.40  # silence between narration clips

# ---------------------------------------------------------------- shot list
# (image, narration clip, split(at seconds|None), camera move, caption, caption style)
SHOTS = [
    ("shot1",  "s1", 0.0,  "in",    "3:00 AM",                              "big"),
    ("shot2",  "s1", 3.4,  "out",   "Every window is dark. Except yours.",  "sub"),
    ("shot8",  "s1", 6.4,  "in",    "You close your eyes. Nothing happens.", "sub"),
    ("shot6",  "s1", 8.8,  "out",   "Not tonight. Not ever.",               "sub"),
    ("shot3",  "s2", 0.0,  "right", "Night one feels like a superpower.",   "sub"),
    ("shot9",  "s2", 3.6,  "in",    "Eight extra hours. Every night.",      "sub"),
    ("shot10", "s2", 6.4,  "left",  "2,920 hours a year.",                  "big"),
    ("shot2",  "s3", 0.0,  "in",    "But nobody warns you about this.",     "sub"),
    ("shot3",  "s3", 3.2,  "in",    "Sleep is not a pause.",                "big"),
    ("shot4",  "s3", 5.6,  "out",   "It is a cleaning crew.",               "big"),
    ("shot5",  "s4", 0.0,  "left",  "No sleep. No rinse.",                  "sub"),
    ("shot5",  "s4", 3.2,  "right", "17 hours awake = 0.5 BAC",             "sub"),
    ("shot8",  "s5", 0.0,  "in",    "Day 3. Micro-sleeping.",               "sub"),
    ("shot7",  "s5", 3.4,  "in",    "Two second blackouts.",                "big"),
    ("shot1",  "s5", 6.0,  "out",   "Waking dreams. It will not obey you.", "sub"),
    ("shot6",  "s6", 0.0,  "out",   "How long could you last?",             "big"),
]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("FAILED: %s\n%s" % (" ".join(cmd[:8]), p.stderr[-3000:]))
    return p


def dur(path):
    p = subprocess.run([FF, "-i", path, "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"time=(\d+):(\d+):([\d.]+)", p.stderr)
    if not m:
        sys.exit("no duration for " + path)
    h, mi, s = m[-1]
    return int(h) * 3600 + int(mi) * 60 + float(s)


# ---------------------------------------------------- 2. narration timeline
clips = {k: os.path.join(ROOT, "audio", k + ".mp3") for k in
         ("s1", "s2", "s3", "s4", "s5", "s6")}
cdur = {k: dur(v) for k, v in clips.items()}
print("durations:", {k: round(v, 2) for k, v in cdur.items()})

# absolute start of each narration clip (clips concatenated with GAP between)
order = ["s1", "s2", "s3", "s4", "s5", "s6"]
start = {}
t = 0.0
for k in order:
    start[k] = t
    t += cdur[k] + GAP
total = t - GAP
print("narration total: %.2fs" % total)

# ------------------------------------------------------ 4-5. shot timeline
segments = []
for i, (img, clip, split, move, cap, style) in enumerate(SHOTS):
    s = start[clip] if split is None else start[clip] + split
    if i + 1 < len(SHOTS):
        nimg, nclip, nsplit, *_ = SHOTS[i + 1]
        e = start[nclip] if nsplit is None else start[nclip] + nsplit
    else:
        e = total + 0.9
    segments.append(dict(img=img, t0=s, t1=e, move=move, cap=cap, style=style))

print("cuts: %d over %.1fs  (%.1f cuts/min)"
      % (len(segments), total, len(segments) / total * 60))


# zoompan expressions: keep motion slow and continuous (Ken Burns)
def motion(move, frames, i):
    step = 0.00045
    if move == "in":
        return f"z='min(1.02+{step}*on,1.14)'", "x='iw/2-(iw/zoom/2)'", "y='ih/2-(ih/zoom/2)'"
    if move == "out":
        return f"z='max(1.14-{step}*on,1.02)'", "x='iw/2-(iw/zoom/2)'", "y='ih/2-(ih/zoom/2)'"
    if move == "right":  # drift left->right
        return ("z='1.12'",
                f"x='(iw-iw/zoom)*(on/{frames})'",
                "y='ih/2-(ih/zoom/2)'")
    if move == "left":
        return ("z='1.12'",
                f"x='(iw-iw/zoom)*(1-on/{frames})'",
                "y='ih/2-(ih/zoom/2)'")
    return "z='1.05'", "x='iw/2-(iw/zoom/2)'", "y='ih/2-(ih/zoom/2)'"


os.makedirs(os.path.join(ROOT, "render"), exist_ok=True)
parts = []
for i, seg in enumerate(segments):
    d = max(seg["t1"] - seg["t0"], 0.5)
    frames = int(round(d * FPS))
    z, x, y = motion(seg["move"], frames, i)
    cap = seg["cap"].replace(":", r"\:").replace("'", r"\'")
    size = 96 if seg["style"] == "big" else 46
    font = FONT_BOLD if seg["style"] == "big" else FONT_SUB
    capfile = os.path.join(ROOT, "render", f"cap{i}.txt")
    open(capfile, "w").write(seg["cap"])
    # caption fade: in over .45s, hold, out over .5s before the cut
    a = (f"if(lt(t,0.45),t/0.45,if(lt(t,{d-0.6:.2f}),1,"
         f"if(lt(t,{d-0.1:.2f}),({d-0.1:.2f}-t)/0.5,0)))")
    vf = (
        f"scale={W*2}:-2,"
        f"zoompan={z}:{x}:{y}:d={frames}:s={W}x{H}:fps={FPS},"
        f"vignette=PI/4.6,"
        f"noise=alls=3:allf=t+u,"
        f"eq=contrast=1.06:saturation=1.04,"
        f"drawtext=fontfile={font}:textfile={capfile}:fontcolor=white:fontsize={size}"
        f":borderw=3:bordercolor=black@0.55:line_spacing=8"
        f":x=(w-text_w)/2:y=h-text_h-140:alpha='{a}'"
    )
    out = os.path.join(ROOT, "render", f"seg{i}.mp4")
    run([FF, "-y", "-loop", "1", "-i", os.path.join(ROOT, "shots", seg["img"] + ".png"),
         "-t", f"{d:.3f}", "-vf", vf, "-r", str(FPS),
         "-c:v", "libx264", "-preset", "medium", "-crf", "21",
         "-pix_fmt", "yuv420p", "-an", out])
    parts.append(out)
    print(f"  seg{i}: {seg['img']} {d:5.2f}s  {seg['move']:5s}  '{seg['cap']}'")

# ------------------------------------------------------ 6-7. audio design
narr_inputs, narr_filters = [], []
for i, k in enumerate(order):
    narr_inputs += ["-i", clips[k]]
    narr_filters.append(f"[{i}:a]aresample=48000,apad=pad_dur={GAP}[n{i}]")
narr_chain = "".join(f"[n{i}]" for i in range(len(order)))
narr_chain += f"concat=n={len(order)}:v=0:a=1[narr]"

cut_times = [s["t0"] for s in segments[1:]]
impacts = []   # (input index, filter)
idx = len(order)
extra_inputs = []
# low drone bed
extra_inputs += ["-f", "lavfi", "-i", f"sine=frequency=55:duration={total+2}"]
drone_idx = idx; idx += 1
extra_inputs += ["-f", "lavfi", "-i", f"sine=frequency=82.4:duration={total+2}"]
drone2_idx = idx; idx += 1
# one noise source, copied via asplit for each impact
impact_n = len(cut_times)
extra_inputs += ["-f", "lavfi", "-i", f"anoisesrc=color=brown:duration={total+2}:amplitude=0.5"]
noise_idx = idx; idx += 1

filters = narr_filters + [narr_chain]
parts_split = "".join(f"[nz{i}]" for i in range(impact_n)) if impact_n else ""
filters.append(f"[{noise_idx}:a]asplit={impact_n}{parts_split}" if impact_n else "")
imp_labels = []
for j, ct in enumerate(cut_times):
    filters.append(
        f"[nz{j}]highpass=f=90,lowpass=f=900,afade=t=in:st=0:d=0.02,"
        f"afade=t=out:st=0.05:d=0.35,volume=0.65,adelay={int(ct*1000)}|{int(ct*1000)}[imp{j}]")
    imp_labels.append(f"[imp{j}]")
filters.append(f"[{drone_idx}:a]volume=0.045,tremolo=f=0.12:d=0.5[dr]")
filters.append(f"[{drone2_idx}:a]volume=0.022,tremolo=f=0.11:d=0.6[dr2]")
mix_in = "[narr]" + "".join(imp_labels) + "[dr][dr2]"
filters.append(f"{mix_in}amix=inputs={2+impact_n+1}:duration=first:dropout_transition=0,"
               f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=out:st={total-0.6:.2f}:d=0.6[aout]")

fc = ";".join(f for f in filters if f)
run([FF, "-y"] + narr_inputs + extra_inputs +
    ["-filter_complex", fc, "-map", "[aout]",
     "-c:a", "aac", "-b:a", "192k", os.path.join(ROOT, "render", "mix.m4a")])

# ---------------------------------------------------------- 8. final render
lst = os.path.join(ROOT, "render", "list.txt")
open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst,
     "-i", os.path.join(ROOT, "render", "mix.m4a"),
     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
     "-movflags", "+faststart", os.path.join(ROOT, "out.mp4")])

print("\nDONE ->", os.path.join(ROOT, "out.mp4"), "%.1fs" % total)
json.dump({"total": total, "cuts": len(segments),
           "segments": segments}, open(os.path.join(ROOT, "render", "timeline.json"), "w"), indent=2)
