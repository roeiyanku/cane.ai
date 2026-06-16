# cane.ai — on-device web app

A single page that runs the whole cane.ai pipeline **on your phone** — camera
in, sidewalk/road/terrain segmentation + depth + obstacle fusion + overlay out —
with no PC and no app install. Both models run in the phone's browser via
[transformers.js](https://github.com/huggingface/transformers.js) on WebGPU.

It mirrors `server/perception.py` (fusion) and `live_view.py` (overlay), ported
to JavaScript.

## Requirements

- **Android Chrome** (recent) with WebGPU. Check `chrome://gpu` shows WebGPU
  enabled. Without WebGPU it falls back to WASM (much slower, but works).
- The page must be served over **HTTPS** — browsers only give camera access in a
  secure context. (`http://192.168.x.x` will *not* get the camera.)

## Getting it on your phone

Pick one:

### A) GitHub Pages (easiest, permanent, fully on-device after load)
1. Put this `webapp/` folder in a GitHub repo.
2. Repo → Settings → Pages → deploy from branch, `/` or `/webapp`.
3. Open the resulting `https://<you>.github.io/<repo>/` on your phone, tap
   **Start camera**.

### B) Quick tunnel (no repo, good for a one-off test)
From this folder on the PC:
```bash
# serve the static files locally
python -m http.server 8080
# in another terminal, expose them over HTTPS
npx cloudflared tunnel --url http://localhost:8080
```
Open the printed `https://...trycloudflare.com` URL on your phone.

> The tunnel only serves the *page*; inference still runs on the phone. After the
> models are cached you can even go offline.

## First run

The first **Start** downloads model weights (~200 MB) and caches them in the
browser. Later runs start fast. Expect a slideshow (~1–4 fps) — a phone GPU is
much weaker than a desktop.

## Tuning

In `index.html`:
- `SEG_MODEL` — `segformer-b0` is light; swap to `b2` for quality at a big speed
  cost.
- `DEPTH_MODEL` — Depth Anything V2 Small.
- `WORK_W` — inference/fusion resolution (default 320). Lower = faster, coarser.

## Limitations

Same safety caveats as the main README — research-licensed models, no real-world
validation, depth is **relative** not metres. This is a prototype, not a guidance
device for unsupervised use.
