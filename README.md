
# cane.ai

**Monocular perception for assistive pedestrian navigation.** A single camera
frame goes in; sidewalk segmentation, relative depth, and fused navigation cues
come out. Both models run locally — no third-party inference API.

- 🎥 **Try it on your phone** in under a minute → [Quick start](#quick-start)
- 🧠 **How it works** → [Pipeline](#pipeline)
- 🖥️ **Run the perception server** → [Server](#server)
- ⚠️ **Read before deploying** → [Safety & licensing](#safety--licensing)

---

## Quick start

The demo is a single page — [`webapp/index.html`](webapp/index.html) — that runs
the entire pipeline live in your phone's browser. No PC server and no app
install; both models execute on-device via transformers.js.

> **The page must be served over HTTPS.** Browsers only grant camera access in a
> secure context, so opening `index.html` directly from disk (`file://`) leaves
> the camera black. Serve it instead:

```bash
cd webapp
python -m http.server 8080                            # 1. serve the page
npx cloudflared tunnel --url http://localhost:8080    # 2. expose it over HTTPS (new terminal)
```

Open the printed `https://….trycloudflare.com` URL on your phone and tap **Start
camera**. The models download once (~200 MB) and cache in the browser; the live
overlay appears after that.

For GitHub Pages hosting, tuning, and device requirements, see
[`webapp/README.md`](webapp/README.md).

---
<img width="1080" height="2400" alt="Screenshot_20260628_183047_Chrome" src="https://github.com/user-attachments/assets/5aec7607-28e7-4034-8974-71c657f68f4d" />

## Pipeline

```
  image ──▶ SegFormer ──────▶ sidewalk / walkable region
        └─▶ Depth Anything ─▶ per-pixel relative depth
                              │
                              ▼
                         fusion ──▶ { walkable region, centerline,
                                      obstacles on path + relative distance }
```

| Stage            | Model               | Output                                            |
| ---------------- | ------------------- | ------------------------------------------------- |
| **Segmentation** | SegFormer           | Sidewalk / walkable region and its boundaries     |
| **Depth**        | Depth Anything V2   | Per-pixel relative depth from a single camera     |
| **Fusion**       | —                   | Nearest obstacle *on the corridor*, with position |

Fusion keeps only the obstacles that sit on the walkable corridor and reports the
nearest one with a relative distance and a left/right position.

---

## Server

```bash
cd pathsense
pip install -r requirements.txt
uvicorn server.api:app --host 0.0.0.0 --port 8000
```

The first request downloads model weights (a few hundred MB) and is slow. A
**GPU is strongly recommended** — on CPU the server is far from real-time.

Test it:

```bash
curl -F "file=@frame.jpg" http://localhost:8000/segment | jq '.nearest_obstacle, .walkable_ratio'
```

### Response shape

```jsonc
{
  "height": 576, "width": 768,
  "walkable_mask": [[0,1,...], ...],   // 1 = sidewalk
  "walkable_ratio": 0.27,
  "obstacle_mask":  [[0,0,...], ...],  // 1 = near object on the path
  "nearest_obstacle": { "distance_rel": 0.9, "column_frac": 0.47 } | null,
  "centerline": [[row, col], ...]      // sampled bottom → top
}
```

`distance_rel` is **relative** (≈1.0 = very close, 0 = far), normalized per
frame. It is not metres — see [Depth is relative](#depth-is-relative).

---

## Clients

| Client                              | Server needed | Notes                                                          |
| ----------------------------------- | ------------- | -------------------------------------------------------------- |
| [`live_view.py`](live_view.py)      | Yes           | Desktop: plays a walking video through `/segment` and overlays the navigation cues. Usage is in the module header. |
| [`webapp/`](webapp/)                | No            | Phone: runs both models on-device in the browser via transformers.js — same overlay, no PC. |

---

## Depth is relative

Depth Anything outputs *relative* inverse-depth. Converting `distance_rel` into
metres requires calibration — e.g. camera intrinsics plus mounting height and a
ground-plane assumption, or a one-time reference object at a known distance.
Until calibrated, treat distance as ordinal ("closer vs. farther"), not metric.

---

## Safety & licensing

This is a research prototype. Read this section before any deployment or
commercial use.

1. **Model licensing is a hard gate.** The default SegFormer checkpoint
   (`nvidia/segformer-b2-finetuned-cityscapes-…`) is trained on Cityscapes, which
   is research / non-commercial. It is a prototyping stand-in only. A shipped
   product must replace it with a model trained on data you own or have licensed;
   confirm the Depth Anything variant's license as well.

2. **This is a safety device.** It is intended to *assist* a capable navigation
   pipeline, not to be the sole guidance for a blind user crossing real streets.
   It has had no real-world safety validation. Do not deploy it with users
   without supervised testing and the appropriate review — liability, insurance,
   and possibly regulatory — which is a matter for qualified counsel, not this
   README.

3. **Fail safe, fail loud.** When perception is uncertain (low `walkable_ratio`,
   conflicting depth), the downstream system should go quiet and signal
   uncertainty rather than confidently guide someone.

---

## Project structure

```
requirements.txt     # Python deps for the server
server/
├── perception.py    # both models + fusion logic
├── api.py           # FastAPI endpoints
└── __init__.py
live_view.py         # desktop client: video → server → overlay
webapp/              # phone client: both models on-device in the browser
```

---

## Authors

Created by **Roei Yanku** and **Tzvi Lengerman**.
