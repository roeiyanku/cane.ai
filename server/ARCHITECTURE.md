# cane.ai — End-to-End Architecture (camera-only)

A **robotic guide-dog** for blind and low-vision users: a motorized, wheeled
device driven by an **NVIDIA Jetson with a single RGB camera** — **no LiDAR**.
It doesn't tell the user where to go (they know the route); it **leads them along
the walkable path and keeps them out of danger**, like a guide dog. The user
holds the handle and follows; the device walks the line, avoids obstacles on the
path, and signals when to stop.

> **Scope note: camera only, at least for now.** Everything below is built around
> one camera. There is no LiDAR and no metric depth. This is deliberate (cost,
> simplicity), and it has real consequences — see **"What camera-only costs you"**
> below. LiDAR is the natural future upgrade, not a current dependency.
>
> The `webapp/` phone app is a **working camera-only demo** of the perception:
> it runs the same two models in a phone browser and shows the path/road
> segmentation and on-path obstacles live from the camera.

---

## ⚠️ Safety is the architecture, not a section

This device **moves a person who cannot see what it is about to do.** With a
single camera (no metric depth), the system is *less certain* than a LiDAR rig
would be, so safety leans even harder on three rules:

1. **Fail safe, fail loud.** When the camera is uncertain (low confidence, glare,
   the path disappears), **slow and stop, signal** — never guess while moving.
2. **Intelligent disobedience.** Like a guide dog, it **refuses** a commanded move
   that heads into the road or an obstacle on the path.
3. **The human is always in control of last resort.** It leads gently, never
   drags; the user can always physically resist or stop it, and a hardware kill
   is within reach.
4. **Go slow.** Because camera-only depth is unreliable for drop-offs (see below),
   conservative speed is a safety feature, not a limitation.

Nothing here is for unsupervised real-world use without staged testing, safety
validation, and the company's own liability/regulatory review.

---

## What the device does (interaction model)

The "leash" model — device leads, user follows — on two timescales:

- **Continuous (device):** lead down the walkable corridor — set heading along the
  path centerline, avoid obstacles **on the path**, slow/stop when the path is
  blocked or unclear. The user just holds the handle and walks; no micro-steering.
- **Discrete (user):** at junctions the user calls the route ("turn left", "cross
  now"). The device executes it **only if safe**, else holds and signals.

No map, no GPS, no global planner — a **local, reactive, shared-control** system.

---

## System overview (camera-only loop, runs on the Jetson)

```
                          ┌─────────────────────────────────────┐
   USER (route intent) ──►│ holds handle + walks; calls turns     │
                          └──────────────┬──────────────────────┘
        SENSING            PERCEPTION     │ turn / go-stop          ACTUATION
  ┌──────────┐        ┌───────────────────▼──────────────┐
  │  Camera  │──RGB──►│ Segmentation → path vs road       │──► wheel motors
  │  (RGB)   │        │ Monocular depth → relative "how    │    (lead / correct,
  └──────────┘        │   far / what rises off the ground" │     never drag)
                      │ → walkable corridor + centerline   │
                      │ → obstacles ON THE PATH only       │──► haptics
                      └───────────────┬───────────────────┘    (turn / stop /
                                      │                          "blocked" cues)
                          ┌───────────────────────────────────┐
                          │ SAFETY SUPERVISOR (overrides all)  │
                          │ E-stop · speed cap · user clutch · │
                          │ confidence gate · go-slow          │
                          └───────────────────────────────────┘
```

---

## Hardware

| Component | Role |
|---|---|
| **Camera (RGB)** | The *only* sensor — sidewalk/road understanding **and** rough depth |
| **NVIDIA Jetson** | Onboard CUDA compute; runs perception + control in real time, offline |
| **Drive base + handle** | Differential-drive wheels + a handle for the user; haptic actuator(s) |
| *(future) LiDAR* | Not present. Would add metric distance + reliable curb detection. |

---

## Layer 1 — Sensing
One RGB camera, mounted to see the path ahead. Output: a stream of frames. That's
it — the whole system is monocular.

## Layer 2 — Perception
Two models on each frame (the same ones the `webapp/` demo and `perception.py`
run):

1. **Semantic segmentation** (SegFormer) — labels each pixel: **walkable path**
   vs **road** vs other. This is the **primary signal** and the make-or-break for
   accuracy. A generic model confuses vegetation/walls for pavement; the fix is
   **retraining on your own sidewalk footage** (Layer 7).
2. **Monocular depth** (Depth Anything) — **relative** inverse-depth (ordinal
   "closer/farther", **not metres**). Used to flag things that **rise above the
   ground plane within the walkable corridor** = obstacles on the path.

**Obstacles only count if they're on the path.** Detections off the walkable
corridor are ignored — a chair on the sidewalk matters; a wall to the side does
not. (`fuse()` already gates obstacles to the walkable band.)

**Output:** the walkable corridor + its centerline (a heading), and the nearest
on-path obstacle (as a relative, ordinal distance).

## Layer 3 — World model
Because there's no metric depth, this stays **lightweight and image-relative**
(not a metric costmap): the walkable corridor, a centerline heading, and on-path
obstacle flags with ordinal "how close." Good enough for local lead-and-avoid.

## Layer 4 — Decision (shared-control law)
Turn perception + user intent into a safe motion command:
1. Default heading = follow the path centerline.
2. Bias by the user's turn intent at junctions.
3. Slow/stop for obstacles on the path.
4. **Intelligent disobedience** — refuse a move into the road or a blocked path.

This is exactly the logic validated in `sim/guide_sim.py` (lead / avoid / stop /
cross), just fed by real perception instead of a drawn world. ROS 2 is a sensible
backbone but you only need a local controller, not global navigation.

## Layer 5 — Safety supervisor & actuation
A supervisor outside the controller, able to override it and the user:
- **E-stop / hard speed cap / bounded deceleration**, and a **go-slow** default.
- **User clutch / override** — the user can always resist or halt.
- **Confidence gate** — low segmentation confidence, lost path, or glare → stop &
  signal.

Gated commands → **wheel motors** (differential drive) and **haptics**
(lead direction / turn / stop / "blocked").

## Layer 7 — Data (the real accuracy lever)
With camera-only, accuracy *is* the segmentation model. Mostly existing data
topped up with **your own sidewalk footage** for the classes generic datasets get
wrong (curb cuts, crosswalks, regional paving, edges). Label with CVAT/Roboflow.

> **Licensing is a hard commercial gate.** The default ADE/Cityscapes checkpoints
> are research-only. Retrain on data you own or have licensed before shipping.

## Layer 8 — Model training
Fine-tune a **real-time** segmentation backbone (DDRNet/PIDNet/BiSeNet, or
SegFormer-B0/B1) on your data; export to **TensorRT** for the Jetson. Evaluate
per-class IoU, watching the safety-critical classes (path edge, road boundary).

---

## What camera-only costs you (be honest about this)

Dropping LiDAR removes two things that matter for a guide dog:

1. **No metric distance.** You get "closer/farther", not "2.4 m". Fine for
   relative lead-and-avoid; you can't reason in real units.
2. **Unreliable curb / drop-off detection.** Spotting where the ground *falls
   away* (curbs down, steps, holes) — the single most important guide-dog
   safety job — is **hard and noisy from a single camera**. Monocular depth
   doesn't give a trustworthy ground plane.

**Mitigations until/unless LiDAR is added:** conservative speed, aggressive
fail-safe (stop when unsure), relying on the user's own residual cues at curbs,
and being explicit that this is a **prototype**, not a validated unsupervised
guide. The `perception.py` LiDAR hooks (`metric_depth_from_sensor`,
`depth_is_metric`) stay as the seam for adding LiDAR later.

---

## What survives in the code / what's new
- **Survives:** `perception.py` (segmentation + monocular depth + fusion) and the
  control logic proven in `sim/guide_sim.py` — both are already camera-only.
- **New work:** robustifying on-path obstacle detection from monocular depth; the
  shared-control law + safety supervisor; actuation (diff-drive + haptics);
  TensorRT for real-time; and above all **retraining segmentation on owned data**.

## Build order
1. **Perception on real footage** — validate path/road segmentation + on-path
   obstacles on recorded walks (the `webapp/` demo + logged video). De-risk
   accuracy first.
2. **Retrain segmentation** on your sidewalk data — the biggest accuracy win.
3. **Control in sim** — the `sim/` logic is the controller; keep tuning it.
4. **Actuation bring-up** — diff-drive + haptics, behind the safety supervisor and
   a hardware E-stop, at crawl speed, supervised.
5. **Real-time** — TensorRT, profiling to hold the loop rate.
6. **Staged real-world testing** — supervised, escalating, with safety review.

## Repository layout (proposed)
```
pathsense/
├── README.md
├── server/ARCHITECTURE.md       ← this file
├── server/perception.py         ← segmentation + monocular depth + fusion
├── server/api.py                ← dev/debug HTTP endpoint
├── webapp/                      ← working camera-only phone demo (in-browser)
├── sim/                         ← guide-dog decision-logic simulation
├── control/                     ← shared-control law + intelligent disobedience
├── safety/                      ← supervisor: E-stop, limits, confidence gate
├── actuation/                   ← diff-drive motor control + haptics
└── train/                       ← dataset prep, fine-tuning, eval, TensorRT export
```

## Open questions
- Which **Jetson** model (sets the real-time budget).
- **Junction intent input** — the control the user uses to call turns.
- **Drive base** — kinematics, hardware E-stop / clutch.
- **Curb safety** — how to handle drop-offs given no LiDAR (speed limits? user
  cues? a cheap dedicated down-facing sensor as a middle ground?).
