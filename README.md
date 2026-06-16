# cane.ai — Local Perception Server

Monocular perception for assistive pedestrian navigation. A single camera frame
goes in; sidewalk segmentation, depth, and fused navigation cues come out. No
third-party inference API — both models run locally.

## What it does

```
  image ──▶ SegFormer ───▶ sidewalk / walkable region
        └─▶ Depth Anything ─▶ per-pixel relative depth
                              │
                              ▼
                         fusion ──▶ { walkable region, centerline,
                                      obstacles on path + relative distance }
```

- **Segmentation** (SegFormer) finds the sidewalk and its boundaries.
- **Depth** (Depth Anything V2) estimates how far things are, from one camera.
- **Fusion** keeps only obstacles that sit *on the walkable corridor* and reports
  the nearest one with a relative distance and left/right position.

## Run it

```bash
cd pathsense
pip install -r requirements.txt
uvicorn server.api:app --host 0.0.0.0 --port 8000
```

First request downloads model weights (a few hundred MB) and is slow.
A **GPU is strongly recommended** — on CPU this is far from real-time.

Test it:

```bash
curl -F "file=@frame.jpg" http://localhost:8000/segment | jq '.nearest_obstacle, .walkable_ratio'
```

## Response shape

```jsonc
{
  "height": 576, "width": 768,
  "walkable_mask": [[0,1,...], ...],   // 1 = sidewalk
  "walkable_ratio": 0.27,
  "obstacle_mask":  [[0,0,...], ...],  // 1 = near object on the path
  "nearest_obstacle": { "distance_rel": 0.9, "column_frac": 0.47 } | null,
  "centerline": [[row, col], ...]      // sampled bottom→top
}
```

`distance_rel` is **relative** (≈1.0 = very close, 0 = far), normalized per frame.
It is NOT metres — see "Depth is relative" below.

## ⚠️ Before any commercial use — read this

1. **Dataset/model licensing is a hard gate.** The default SegFormer checkpoint
   (`nvidia/segformer-b2-finetuned-cityscapes-...`) is trained on **Cityscapes,
   which is research / non-commercial.** It is a prototyping stand-in ONLY.
   For a shipped product, replace it with a model trained on data you own or
   have licensed. Confirm the Depth Anything variant's license too.
2. **This is a safety device.** It is intended to *assist* a sighted-capable
   pipeline, not to be the sole guidance for a blind user crossing real streets.
   It has had **no real-world safety validation.** Don't deploy it with users
   without supervised testing and proper review (liability, insurance, possibly
   regulatory) — that's for the company's counsel, not this README.
3. **Fail safe, fail loud.** When perception is uncertain (low walkable_ratio,
   conflicting depth), the downstream system should go quiet and signal
   uncertainty rather than confidently guide someone.

## Depth is relative, not metric

Depth Anything outputs *relative* inverse-depth. To turn `distance_rel` into
metres you need calibration — e.g. camera intrinsics + mounting height and a
ground-plane assumption, or a one-time reference object at known distance. Until
then, treat distance as ordinal: "closer vs. farther," not "2.4 m."

## Clients

Two ways to see the perception output:

- **Desktop (uses this server):** [`live_view.py`](live_view.py) plays a walking
  video through `/segment` and draws the navigation overlay. Usage is in its
  module header.
- **Phone (no server):** [`webapp/`](webapp/) runs both models fully on-device in
  the browser via transformers.js — same overlay, no PC needed. See
  [`webapp/README.md`](webapp/README.md).

## Files

```
requirements.txt     # Python deps for the server
server/
├── perception.py    # both models + fusion logic
├── api.py           # FastAPI endpoints
└── __init__.py
live_view.py         # desktop client: video → server → overlay
webapp/              # phone client: both models on-device in the browser
```
