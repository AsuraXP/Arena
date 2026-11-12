# v2 — fully procedural animation (no source images)

`demo3/build3.py` renders every frame in code. There are **no stills, no
generated art, no stock footage** — only the narration audio is pre-recorded.

## What is actually animated

| Scene | Motion |
|---|---|
| CITY | 3 parallax skyline layers, drifting clouds, stars, ~700 windows switching off in a ripple, one window burning |
| STREET | facade bricks, lit window with a **jointed figure waving** (shoulder+elbow swing), breathing light, wet-asphalt reflection |
| BEDROOM | blanket **breathing**, eyes **blinking**, 7-segment clock, moonbeam through the window |
| ROOM | **walk cycle** (hip/knee/shoulder/elbow), then seated typing animation under a lamp |
| OCEAN | **pulsing jellyfish** with 9 undulating tentacles each, god rays, rising bubbles, drifting motes |
| LAB | spinning dials, live **oscilloscope waveform**, gurney, blinking subject eye, observers |
| CAMERA | every shot dollies/zooms/pans + handheld shake |

## The rig

`person()` draws a silhouette from a skeleton: hip → knee → ankle, shoulder →
elbow → hand, torso polygon, head. Angles are in radians with `0 = down`,
`+π/2 = out to the side`. Poses are pure functions of time, so motion is
continuous and deterministic:

```
walk_pose(phase)  → thigh/shin/arm/fore angles from sin(phase)
wave_pose(t)      → arm raised, forearm swinging
sit_pose(t)       → seated, hands tapping
```

## Camera

`Cam(x, y, z)` maps world→screen. Every shot has a start and end camera, and
`Cam.blend` eases between them, so you get a real dolly/zoom rather than a
static frame. World units are screen pixels with the origin at frame centre.

## Render

    cd demo3 && python3 build3.py      # ~13 min for 79s

Re-render only some shots:

    ONLY_SCENES=city,lab python3 build3.py

Output: 1920x1080 30 fps H.264, 22 shots over 78.5s (16.8 shots/min).
