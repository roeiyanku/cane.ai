# cane.ai — End-to-End Architecture

A **robotic guide-dog** for blind and low-vision users: a motorized, wheeled
device with an onboard camera and LiDAR, driven by an NVIDIA Jetson. It does not
tell the user where to go — the user already knows the route. It **physically
leads them along the walkable path and keeps them out of danger**, exactly like a
guide dog: the user holds the handle and follows; the device walks the line,
avoids obstacles, stops at curbs and roads, and refuses any move that would hurt
them.

> **This supersedes the earlier "phone app + audio cues + GPS" architecture.**
> The product is not a phone app and not a turn-by-turn navigator. It is an
> actuated assistive robot running on embedded hardware. The data/training and
> licensing material from the old doc survives (Layers 7–8 below); everything
> about the runtime is new.

---

## ⚠️ Safety is the architecture, not a section

This device **moves a person who cannot see what it is about to do.** A wrong
action can put someone into traffic or off a curb. Three consequences shape every
layer:

1. **Fail safe, fail loud.** When any sensor drops, a frame goes stale, depth is
   sparse, or perception confidence is low → **stop and signal**. Never guess
   while in motion.
2. **Intelligent disobedience.** Like a trained guide dog, the device **refuses**
   a user's commanded turn if it leads into a road, an obstacle, or a drop-off.
   The safety supervisor can override the user *and* the planner.
3. **The human is always in control of last resort.** The device leads gently; it
   must never drag. The user can always physically resist or stop it, and a
   hardware kill is within reach.

Nothing here is for unsupervised real-world use without staged testing, safety
validation, and the company's own liability/regulatory review.

---

## What the device does (interaction model)

The "leash" model — device leads, user follows — split into two timescales:

- **Continuous (the device's job).** Lead down the walkable corridor: set heading
  along the sidewalk centerline, avoid obstacles, automatically slow and stop at
  drop-offs and road edges. The user just holds the handle and walks — no
  micro-steering.
- **Discrete (the user's job).** At junctions, the user issues the route decision
  ("turn left," "cross now") via a simple control. The device executes it **only
  if safe**, otherwise holds position and signals. The user supplies the *where*;
  the device supplies the *safe how*.

Because the user owns the route, the system needs **no map, no GPS, no SLAM, no
global path planner.** It is a **local, reactive, shared-control** system.

---

## System overview (the control loop, ~10 Hz)

```
                          ┌─────────────────────────────────────┐
   USER (route intent) ──►│ holds handle + walks; calls turns     │
                          │ at junctions ("left", "cross")        │
                          └──────────────┬──────────────────────┘
        SENSING            WORLD MODEL    │ desired turn / go-stop      ACTUATION
  Camera ─► seg ─────┐                    ▼
                     ├─► local BEV ──► ┌─────────────────────────────┐
  LiDAR ─► geometry  │   costmap       │ Shared-control law:           │─► wheel motors
        ├ rising  → obstacle (lethal)  │  • lead along walkable        │   (lead/correct,
        ├ falling → drop-off  (lethal) │    centerline                 │    never drag)
        └ road (seg)        → lethal   │  • bias by user turn intent   │
                     │                 │  • INTELLIGENT DISOBEDIENCE   │─► haptics
  IMU + encoders ─► odometry           │    (veto unsafe moves)        │   (turn / stop /
                     └─────────────────┴───────────┬───────────────────┘    "blocked" cues)
                                                   │
                          ┌──────────────────────────────────────────┐
                          │ SAFETY SUPERVISOR (authority over all)     │
                          │ E-stop · speed cap · decel limit · user    │
                          │ clutch · sensor watchdog · confidence gate │
                          └──────────────────────────────────────────┘
```

---

## Hardware

| Component | Role |
|---|---|
| **Camera (RGB)** | Semantic understanding — what is walkable (sidewalk / path / crossing) vs road |
| **LiDAR** | Metric 3D geometry — obstacles, ground plane, **drop-offs**; the source of real distances |
| **IMU + wheel encoders** | Odometry — keeps the local costmap stable as the device moves |
| **NVIDIA Jetson** | Onboard CUDA compute; runs perception + control in real time, fully offline |
| **Drive base + handle** | Differential-drive wheels (lead/steer) + handle for the user; haptic actuator(s) |

The specific Jetson (Orin Nano/NX vs AGX Orin vs Xavier) sets the optimization
budget: FP16 on AGX-class, INT8 + a lighter segmentation model on Nano-class.

---

## Layer 1 — Sensing & time sync

Produce, every cycle, a **synchronized** bundle: one RGB frame, one LiDAR scan,
and the latest odometry, matched by timestamp (or hardware trigger). For a moving
pedestrian, a few hundred ms of camera/LiDAR skew puts obstacles in the wrong
place — sync is a correctness requirement, not a nicety.

**Output:** time-aligned `(rgb, point_cloud, odom)` tuples.

---

## Layer 2 — Calibration & metric depth

One-time **calibration** of camera intrinsics (K) and the LiDAR→camera extrinsics
(R, t). At runtime, **project the LiDAR point cloud into the camera image** to
build a metric depth map aligned to the RGB frame.

This is the layer that lights up the dormant hooks already in
[`perception.py`](perception.py): `metric_depth_from_sensor()`, the
`depth_is_metric` flag, `GROUND_MARGIN_M`, and `nearest_obstacle.distance_m`. The
LiDAR replaces Depth Anything's *relative* inverse-depth as the source of truth;
monocular depth survives only as an optional densifier when LiDAR returns are
sparse.

**Output:** a metric (metres) depth map aligned to the camera frame.

---

## Layer 3 — Perception

Two parallel branches fused into a geometric understanding of the scene:

1. **Semantic segmentation (camera).** What is walkable surface vs **road**. The
   road/sidewalk boundary is a *safety* boundary here, not cosmetic. For
   real-time on Jetson this must be a light model (DDRNet / PIDNet / BiSeNet, or
   at most SegFormer-B0) **exported to TensorRT**. See licensing gate in Layer 7.
2. **Geometry (LiDAR).** A **RANSAC ground-plane fit** on the point cloud, then:
   - **Positive obstacles** — points rising above the ground plane (poles,
     people, bollards). This is the existing
     [`ground_plane_obstacles`](perception.py) logic, made truly 3D.
   - **Negative obstacles** — where the ground **falls away**: curbs down,
     descending stairs, holes. **This is new and is the single most important
     life-safety feature** — the classic thing a guide dog exists to prevent.
     The current code only flags rising geometry; a falling-ground branch must be
     added.

**Output:** classified scene geometry — walkable surface, road, positive
obstacles, negative obstacles — all in metric coordinates.

---

## Layer 4 — World model (BEV costmap)

Project perception into a small **top-down (bird's-eye) costmap** around the
device — the representation a planner actually consumes (planners think in the
ground plane, not in pixels):

- walkable corridor → low cost
- everything else off-corridor → high cost
- positive/negative obstacles and **road** → lethal (inflated by a safety margin)

**Output:** a rolling local costmap + the walkable centerline as a candidate path.

---

## Layer 5 — Decision (shared-control law)

Turn the world model + user intent into a **safe velocity command**:

1. Default heading = follow the walkable centerline.
2. Bias by the user's discrete turn intent at junctions.
3. Avoid lethal cells (local reactive avoidance).
4. **Intelligent disobedience** — if the resulting motion would enter a road,
   obstacle, or drop-off, refuse it: hold, slow, or stop, and emit a "blocked"
   cue instead.

ROS 2 is a sensible backbone for wiring this, but you need only the **local
costmap + a custom controller** — not Nav2's global navigation machinery.

**Output:** a desired `(linear, angular)` velocity, pre-safety-gate.

---

## Layer 6 — Safety supervisor & actuation

A supervisor **outside** the planner, with authority to override it and the user:

- **E-stop** on any lethal-range obstacle/edge, regardless of intent.
- **Hard speed cap** at walking pace; bounded deceleration (no jerk).
- **User clutch / override** — the user can always resist or halt; the device
  assists, never drags.
- **Sensor watchdog & confidence gate** — camera/LiDAR dropout, stale frame,
  diverging odometry, sparse LiDAR, or low perception confidence → stop & signal.

Gated commands go to:
- **Wheel motors** (differential-drive kinematics → left/right velocities).
- **Haptics** — encode lead direction / turn / stop / "blocked" as vibration.

**Output:** safe motor commands + haptic cues to the user.

---

## Layer 7 — Data (training the segmentation model)

*(Carried forward from the previous architecture — still valid.)* The model is
only as good as its data: **mostly existing data, topped up with your own footage**
for accessibility-specific classes generic datasets handle poorly.

| Dataset | What it gives you | License notes |
|---|---|---|
| **Cityscapes** | Urban scenes, sidewalk/road/person/car | Research/non-commercial — **gate before selling** |
| **Mapillary Vistas** | Large, diverse global street imagery | Research license; commercial tier exists |
| **SideGuide** | Purpose-built for sidewalk navigation | Verify current license |
| **ADE20K** | Broad scene parsing, good for pretraining | Permissive-ish; verify |

Your own footage is the differentiator — curb cuts/dropped curbs, crosswalks,
surface hazards, regional sidewalk materials and edge types. Label with CVAT or
Roboflow → pixel masks.

> **Dataset/model licensing is a hard commercial gate.** The default checkpoints
> in `perception.py` are research-only Cityscapes/ADE stand-ins. Before shipping,
> retrain on data you own or have licensed.

---

## Layer 8 — Model training

*(Carried forward.)* **Fine-tune, don't train from scratch.** A real-time
segmentation backbone adapted to your classes gives the best quality for the
least compute.

- **Architecture:** prioritize **real-time** (DDRNet/PIDNet/BiSeNet, or
  SegFormer-B0) — this runs in a control loop on an edge device, not offline.
- **Framework:** PyTorch + Hugging Face `transformers` (you train/serve it
  yourself; no hosted API), then **export to TensorRT** for the Jetson.
- **Evaluation:** mean IoU **per class**, watching safety-critical classes
  (sidewalk edge, curb cut, road boundary) — a good average can hide a deadly
  edge-case failure.

**Output:** a TensorRT-optimized segmentation engine + a per-class eval report.

---

## What survives from the current code vs. what's new

**Survives:** the perception core — camera→walkable segmentation, LiDAR→metric
geometry, ground-plane obstacle fusion. The `Perception` class and `fuse()` are
the right seam; the `depth_is_metric` / `metric_depth_from_sensor` hooks were
written for exactly this hardware.

**New work, roughly in priority order:**
1. **Negative-obstacle (drop-off) detection** in perception — life-safety.
2. **Calibration + LiDAR→metric-depth** layer (lights up `depth_is_metric`).
3. **BEV costmap** output (replace image-mask output for the planner).
4. **Shared-control law + intelligent disobedience.**
5. **Safety supervisor** (E-stop, speed/decel limits, watchdog, clutch).
6. **Actuation** — diff-drive motor control + haptics.
7. **Real-time** — TensorRT segmentation, pipeline parallelism on the Jetson.

---

## Build order (recommended)

1. **Perception upgrade** — add negative-obstacle detection; wire LiDAR metric
   depth (`depth_is_metric=True`). Validate on logged sensor data, off-vehicle.
2. **World model** — project to a BEV costmap; visualize it for debugging.
3. **Control in simulation** — shared-control law + safety supervisor against
   recorded/sim data before touching motors.
4. **Bring up actuation** — diff-drive + haptics, with the safety supervisor and
   a hardware E-stop from day one, at crawl speed, supervised.
5. **Real-time optimization** — TensorRT, profiling to hold the loop rate.
6. **Staged real-world testing** — supervised, escalating, with safety/liability
   review. The value and the risk both live here.

---

## Repository layout (proposed)

```
pathsense/
├── README.md
├── server/ARCHITECTURE.md       ← this file
├── sensors/                     ← camera + lidar sources, time sync
├── calib/                       ← intrinsics/extrinsics, cloud→image, metric depth
├── perception/                  ← seg (TensorRT) + 3D ground plane + obstacles
│   └── perception.py            ← evolve the existing module here
├── worldmodel/                  ← BEV costmap builder
├── control/                     ← shared-control law + intelligent disobedience
├── safety/                      ← supervisor: E-stop, limits, watchdog, clutch
├── actuation/                   ← diff-drive motor control + haptics
├── runtime/                     ← the on-device service loop (Jetson)
├── train/                       ← dataset prep, fine-tuning, eval, TensorRT export
└── server/api.py                ← kept as a dev/debug entry point only
```

---

## Open questions (to confirm with the company)

- **Jetson model** — sets FP16 vs INT8 and how light the segmentation model must
  be.
- **LiDAR output format** — aligned depth image vs raw point cloud, and whether it
  ships pre-calibrated to the camera (decides how much of Layer 2 you build).
- **Junction intent input** — the simple control the user uses to call turns.
- **Drive base** — differential drive assumed; confirm kinematics and the
  hardware E-stop / clutch mechanism.
