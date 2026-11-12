# Hero shot — the craft gap, demonstrated

`hero-shot.mp4` (8s, 1920x1080) is the **same street scene** as demo3, rebuilt with
the things that separate "code animation" from "looks expensive".

## What changed vs demo3

| | demo3 | demo4 |
|---|---|---|
| Framing | wide, window ~15% of frame | **close-up**, window ~50%, SCALE 2.05 |
| Depth | flat | 3 layers + **depth-of-field** blur on bg and fg |
| Light | flat glow | warm interior falloff, **volumetric cone** onto the street, rim light, projected wall shadow |
| Arm motion | pure sine | **anticipation → ease-out-back overshoot → settle** |
| Joint timing | all in sync | **overlapping action**: shoulder leads, forearm lags 4 frames, hand 6 |
| Secondary | none | weight shift, breathing, head tilt, torso counter-rotation |
| Blur | none | **3-sample motion blur** |
| Grade | none | lifted blue shadows, warm highlights, S-curve, grain |

## The two things that mattered most

1. **Framing.** demo3 shot the whole facade so the action was a speck. Cinema is
   mostly about how much of the frame the subject occupies.
2. **Overlapping action.** Look at `wave_pose()`: the shoulder, forearm and hand use
   the *same* sine but sampled at `t`, `t-0.13` and `t-0.20`. That single change is
   what stops joints looking mechanical.

Run: `python3 hd.py` (~14 min: 3 temporal samples per frame for motion blur).
