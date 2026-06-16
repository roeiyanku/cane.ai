"""
cane.ai — perception models.

Two monocular networks running on a single camera frame:
  1. SegFormer  -> semantic segmentation (find the sidewalk / walkable region)
  2. Depth Anything V2 -> per-pixel relative depth (how far things are)

The fusion step combines them into navigation-relevant outputs:
  - the walkable sidewalk region and its boundaries
  - obstacles sitting ON the walkable path, with a relative-distance estimate

IMPORTANT (commercial use):
  The default SegFormer checkpoint below is fine-tuned on Cityscapes, which is
  a RESEARCH / NON-COMMERCIAL dataset. It is a stand-in for prototyping only.
  Before shipping commercially, replace MODEL_SEG with a checkpoint trained on
  data you own or have licensed. Depth Anything V2 (small) is more permissive,
  but confirm the exact variant's license for your use.
"""

from __future__ import annotations
import numpy as np
import torch
from PIL import Image

# ---- model identifiers (downloaded on first run) ----
# ADE20K recognizes paver/tiled sidewalks far better than the Cityscapes model
# (which mislabeled them as "road"). Weights are still non-commercial research
# only — a prototype stand-in. See the licensing note above before shipping.
MODEL_SEG = "nvidia/segformer-b2-finetuned-ade-512-512"
MODEL_DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"

# Fallback class indices (Cityscapes order). The real indices are looked up by
# NAME from the model's own label map at load time — see Perception.__init__ —
# so this keeps working if you swap checkpoints.
SIDEWALK_CLASS = 1
ROAD_CLASS = 0

# An obstacle is a pixel that rises ABOVE the ground plane — i.e. is noticeably
# closer to the camera than the ground at the same image row. The "how much
# closer" margin has different units depending on the depth source:
#   - relative depth (Depth Anything): unitless 0..1  -> GROUND_MARGIN_REL
#   - metric depth   (LiDAR / ARKit):  metres         -> GROUND_MARGIN_M
GROUND_MARGIN_REL = 0.12
GROUND_MARGIN_M = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Perception:
    """Loads both models once and runs the combined perception pass."""

    def __init__(self, seg_model: str = MODEL_SEG, depth_model: str = MODEL_DEPTH, transformers=None):
        # Imported here so the module imports even before transformers pulls weights.
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
            AutoImageProcessor,
            AutoModelForDepthEstimation,
        )

        self.seg_proc = SegformerImageProcessor.from_pretrained(seg_model)
        self.seg = SegformerForSemanticSegmentation.from_pretrained(seg_model).to(DEVICE).eval()

        self.depth_proc = AutoImageProcessor.from_pretrained(depth_model)
        self.depth = AutoModelForDepthEstimation.from_pretrained(depth_model).to(DEVICE).eval()

        # Find the walkable ("sidewalk"/"pavement") and "road" class indices in
        # whatever label set this checkpoint uses, by matching the label names.
        id2label = {int(k): v.lower() for k, v in self.seg.config.id2label.items()}
        self.sidewalk_class = next(
            (i for i, n in id2label.items() if "sidewalk" in n or "pavement" in n),
            SIDEWALK_CLASS,
        )
        self.road_class = next(
            (i for i, n in id2label.items() if "road" in n), ROAD_CLASS
        )
        # Terrain = natural ground beside the path (grass, earth, dirt, sand...).
        terrain_words = ("grass", "earth", "ground", "terrain", "field", "sand", "dirt", "soil")
        self.terrain_classes = [
            i for i, n in id2label.items() if any(w in n for w in terrain_words)
        ]
        print(f"[perception] sidewalk={self.sidewalk_class}, road={self.road_class}, "
              f"terrain={self.terrain_classes}")

        # Depth Anything returns RELATIVE depth. A metric sensor (LiDAR/ARKit)
        # would set this True so the ground-plane detector uses metres. See
        # metric_depth_from_sensor() below.
        self.depth_is_metric = False

    @torch.inference_mode()
    def segment(self, img: Image.Image) -> np.ndarray:
        """Return an (H, W) array of class indices at the image's native size."""
        inputs = self.seg_proc(images=img, return_tensors="pt").to(DEVICE)
        logits = self.seg(**inputs).logits  # (1, C, h/4, w/4)
        upsampled = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear", align_corners=False
        )
        return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    @torch.inference_mode()
    def depth_map(self, img: Image.Image) -> np.ndarray:
        """Return an (H, W) float array of RELATIVE depth, normalized to [0,1].

        Depth Anything outputs inverse-depth-like values where LARGER = CLOSER.
        We normalize per-frame; this is relative, not metric. Converting to
        metres needs calibration (see notes in README).
        """
        inputs = self.depth_proc(images=img, return_tensors="pt").to(DEVICE)
        pred = self.depth(**inputs).predicted_depth  # (1, h, w)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False
        )[0, 0]
        d = pred.cpu().numpy().astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        return d  # 0 = far, 1 = near

    # --- LiDAR / metric-depth extension point --------------------------------
    # Phones with a depth sensor (e.g. iPhone Pro LiDAR via ARKit) can supply a
    # METRIC depth map aligned to the RGB frame. Wiring one in here lets the
    # ground-plane detector use a real height threshold in METRES and lets
    # nearest_obstacle report a true distance. To enable it:
    #   1. produce a metric depth array, same (H, W) as the frame, where the
    #      value is distance in metres and LARGER = FARTHER;
    #   2. return it from this method instead of calling depth_map();
    #   3. set self.depth_is_metric = True.
    # The fusion/obstacle math already branches on depth_is_metric, so nothing
    # downstream needs to change.
    def metric_depth_from_sensor(self, sensor_depth) -> np.ndarray:
        raise NotImplementedError(
            "Plug a LiDAR/ARKit metric depth source here (see comment above)."
        )

    def perceive(self, img: Image.Image) -> dict:
        """Run both models and fuse into navigation outputs."""
        seg = self.segment(img)
        depth = self.depth_map(img)   # or metric_depth_from_sensor(...) when available
        return fuse(seg, depth, sidewalk_class=self.sidewalk_class,
                    road_class=self.road_class, terrain_classes=self.terrain_classes,
                    depth_is_metric=self.depth_is_metric)


def estimate_ground_profile(depth: np.ndarray, ground_mask: np.ndarray,
                            min_pixels: int = 8) -> np.ndarray | None:
    """Expected ground depth for each image row.

    For each row we take the median depth of the pixels segmentation calls
    ground (sidewalk / road / terrain). Rows without enough ground pixels are
    filled by interpolation, then the profile is lightly smoothed.

    This is a monocular stand-in for a true 3D ground-plane fit (compare the
    stereo "v-disparity" method): it anchors "where the ground is" at each row
    so we can flag whatever rises above it. Returns None if there isn't enough
    visible ground to anchor on.
    """
    h = depth.shape[0]
    profile = np.full(h, np.nan, dtype=np.float32)
    for r in range(h):
        vals = depth[r][ground_mask[r] > 0]
        if vals.size >= min_pixels:
            profile[r] = np.median(vals)

    rows = np.arange(h)
    known = ~np.isnan(profile)
    if known.sum() < 2:
        return None
    profile = np.interp(rows, rows[known], profile[known]).astype(np.float32)
    k = max(3, h // 60)                      # light vertical smoothing
    profile = np.convolve(profile, np.ones(k) / k, mode="same").astype(np.float32)
    return profile


def ground_plane_obstacles(depth: np.ndarray, ground_mask: np.ndarray,
                           depth_is_metric: bool,
                           margin_rel: float = GROUND_MARGIN_REL,
                           margin_m: float = GROUND_MARGIN_M) -> np.ndarray:
    """Flag pixels that rise above the ground plane (geometry, not class).

    A pixel is an obstacle if it is noticeably CLOSER to the camera than the
    ground would be at that same row — i.e. something sticking up out of the
    ground. This catches unlabeled objects too, because it uses depth geometry
    rather than the object's semantic class.
    """
    h, w = depth.shape
    profile = estimate_ground_profile(depth, ground_mask)
    obstacle = np.zeros((h, w), dtype=np.uint8)
    if profile is None:
        return obstacle

    prof = profile[:, None]                  # (h, 1), broadcasts across columns
    if depth_is_metric:
        # metric: larger = farther; an obstacle is closer than the ground
        rising = depth < (prof - margin_m)
    else:
        # relative inverse depth: larger = nearer; obstacle is nearer than ground
        rising = depth > (prof + margin_rel)
    obstacle[rising & (ground_mask == 0)] = 1
    return obstacle


def fuse(seg: np.ndarray, depth: np.ndarray,
         sidewalk_class: int = SIDEWALK_CLASS, road_class: int = ROAD_CLASS,
         terrain_classes: list | None = None, depth_is_metric: bool = False) -> dict:
    """Combine segmentation + depth into navigation-relevant structures.

    Returns a JSON-serializable dict:
      walkable_mask : (H,W) uint8  — 1 where sidewalk
      walkable_ratio: float        — fraction of frame that is sidewalk
      road_mask     : (H,W) uint8  — 1 where road
      terrain_mask  : (H,W) uint8  — 1 where natural ground (grass/earth/...)
      obstacle_mask : (H,W) uint8  — 1 where something rises above the ground
                                     plane within the walkable corridor
      nearest_obstacle: {distance_rel, distance_m, column_frac} | None
                        (distance_m is filled only with a metric depth source)
      centerline   : list[[row, col]] sampled down the walkable region
    """
    h, w = seg.shape
    walkable = (seg == sidewalk_class).astype(np.uint8)
    road = (seg == road_class).astype(np.uint8)
    terrain = (np.isin(seg, terrain_classes).astype(np.uint8)
               if terrain_classes else np.zeros_like(walkable))

    # Geometric obstacle detection: flag whatever rises above the ground plane
    # (ground = sidewalk + road + terrain), regardless of its semantic class.
    ground = ((walkable | road | terrain) > 0).astype(np.uint8)
    obstacle = ground_plane_obstacles(depth, ground, depth_is_metric)

    # Restrict obstacles to roughly the walkable corridor: for each row, keep the
    # horizontal band spanned by sidewalk pixels (a simple corridor gate).
    corridor = np.zeros_like(walkable)
    for r in range(h):
        cols = np.where(walkable[r] > 0)[0]
        if cols.size:
            corridor[r, cols.min():cols.max() + 1] = 1
    obstacle = (obstacle & corridor).astype(np.uint8)

    # Centerline: per row, the mean column of walkable pixels (bottom→up sample).
    centerline = []
    for r in range(h - 1, -1, -max(1, h // 40)):
        cols = np.where(walkable[r] > 0)[0]
        if cols.size:
            centerline.append([int(r), int(cols.mean())])

    # Nearest obstacle within the corridor.
    nearest = None
    if obstacle.any():
        ys, xs = np.where(obstacle > 0)
        depths = depth[ys, xs]
        # closest = smallest metres (metric) or largest inverse-depth (relative)
        i = int(np.argmin(depths)) if depth_is_metric else int(np.argmax(depths))
        nearest = {
            "distance_rel": None if depth_is_metric else round(float(depths[i]), 3),
            "distance_m": round(float(depths[i]), 2) if depth_is_metric else None,
            "column_frac": round(float(xs[i]) / w, 3),    # 0 left .. 1 right
        }

    return {
        "height": h,
        "width": w,
        "walkable_mask": walkable.tolist(),
        "walkable_ratio": round(float(walkable.mean()), 4),
        "road_mask": road.tolist(),
        "terrain_mask": terrain.tolist(),
        "obstacle_mask": obstacle.tolist(),
        "nearest_obstacle": nearest,
        "centerline": centerline,
    }
