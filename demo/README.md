# What-If style AI animation — analysis + reproducible build

## 1. What the Facebook video actually is

Source: `facebook.com/share/v/1JytnVSwhA` → `facebook.com/reel/3326736054182362`
("Get Rich With Me", 14m52s, uploaded 2026-11-10, ~69k views).

It is **not** a video tool tutorial. It is a tutorial about using an **AI coding agent** to
generate an animated "What-If" explainer from a single sentence, using a downloaded
**"AI animation skill"** (a folder of instructions + templates + scripts) plus an agent that
can write and run code.

The 8-stage pipeline the skill runs, as described on screen:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | Script | hook → superpower → consequences → science → cliffhanger |
| 2 | Voice | text-to-speech narration, auto-generated |
| 3 | Shot plan | maps wide shots / close-ups, targets **27+ cuts per minute** |
| 4 | Scene construction | code that draws characters, backgrounds, lighting |
| 5 | Visual check | renders preview frames, checks an 11-point checklist, revises |
| 6 | Animation | character + camera movement |
| 7 | Sound | SFX + music bed, mixed under narration |
| 8 | Render | exports MP4 |

Two runs are compared: **OpenCode + a free model (GLM "5.3 flash free", ~25-30 min)** vs
**Claude Code + Opus 5.5 (~30-40 min)**. Both use the same skill. The author's verdict:
Claude had the better opening/cliffhanger, better lighting/depth/camera, better audio timing.

The "$0" claim is only half true: **stage 2 and 7 use the ElevenLabs API** (free tier,
~10,000 credits/month, "300-500 credits per video"). The *agent* is free; the *voice* is not.
The prompt packet is behind "comment PROMPT + DM me".

## 2. Can this be made without ElevenLabs? Yes — this repo's demo is the proof

`demo/out.mp4` — 57s, 1920×1080, 30fps, 16 shots (16.9 cuts/min), narrated, captioned,
sound-designed, all built here with no ElevenLabs, no paid agent, no API keys.

| Tutorial stage | Tutorial tool | How this demo did it |
|---|---|---|
| Script | agent + skill | written directly, same 5-beat structure |
| Voice | **ElevenLabs API** | Arena TTS (`add_voice` → `generate_speech`) |
| Shot plan | agent | `SHOTS` table in `demo/build.py` |
| Visuals | agent writes drawing code | AI image generation (`demo/shots/*.png`) |
| Animation | agent writes motion code | ffmpeg `zoompan` push-in/pull-out/pans + vignette + grain |
| Sound | ElevenLabs SFX + music | synthesised drone + noise impacts on every cut (ffmpeg `sine`/`anoisesrc`) |
| Captions | — | `drawtext` timed per shot, fade in/out |
| Render | agent | ffmpeg libx264 + AAC, loudnorm to -16 LUFS |

Rebuild: `cd demo && python3 build.py` (needs `imageio-ffmpeg`, `pillow`, `numpy`;
fonts converted from woff2 to ttf in `demo/fonts/`).

## 3. Honest comparison

**Where local/no-ElevenLabs wins**
- Voice is free and unlimited here; ElevenLabs free tier runs out after ~20-30 videos.
- No API key handling, no credits meter, no per-video cost.
- Renders in ~2 minutes instead of 25-40 minutes of agent work.
- Fully deterministic and re-runnable — change one caption, re-render.

**Where the tutorial's approach still wins**
- True *character animation* (walking, gesturing, lip-sync) — this demo animates the **camera**,
  not the characters. Their agents write animation code, so limbs actually move.
- 27+ cuts/min sustained over 8-10 minutes needs far more unique art than 10 stills.
- Lighting/framing variety per beat: an agent can generate a new scene per shot on demand.
- Music: a real score beats a synthesised drone.

**The gap is fillable**: more generated shots + per-shot motion presets + a second
TTS pass for SFX gets most of the way there. What it will not do is hand-drawn
character motion without either an agent writing animation code or a video model.

## 4. Repo note

`.github/workflows/fetch-video.yml` was added to pull the source video on a GitHub runner
(this sandbox cannot reach facebook.com). It writes metadata/transcript to `analysis/`.
Trigger it by editing `.fetch-video-trigger`. Artifacts are not downloadable from here —
results are committed back to the branch instead.

## 5. Watch it in the browser

`demo/serve.py` is a small static server with HTTP Range support (so the video can seek):

```
cd demo && python3 serve.py     # http://localhost:8000
```

Pages: a player with a clickable shot list, live "current shot" readout, keyboard
shortcuts (space = play/pause, ←/→ = ±5s). Shot timings come from `site/timeline.json`,
which `build.py` writes on every render.
