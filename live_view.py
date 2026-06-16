"""
cane.ai — live viewer (client) for walking videos.

Plays a walking video through the perception server one sampled frame at a time
and draws a clean navigation overlay: filled walkable/terrain regions, a legend,
and a bold arrow up the path. This does NOT change the server; it just calls
/segment in a loop. The server runs on CPU here, so it's a slow slideshow.

Setup (once):
    pip install opencv-python requests numpy

Run:
    python live_view.py "C:\\Users\\Elad\\Desktop\\Walking Video 2026-05-31 At 23.35.53.mp4"
    python live_view.py "walk.mp4" 30      # process every 30th frame (faster)

Press 'q' in the window to quit.
"""

import sys
import cv2
import numpy as np
import requests

SERVER = "http://localhost:8000/segment"

# Colors in BGR (OpenCV order).
GREEN = (60, 200, 60)     # sidewalk
CYAN = (220, 190, 50)     # terrain
ORANGE = (0, 140, 255)    # direction arrow
RED = (60, 60, 230)       # obstacle


def clean(mask):
    """Fill small holes/specks so the filled region looks solid."""
    k = np.ones((7, 7), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def fill_region(frame, mask, color, alpha=0.45):
    """Blend a translucent solid color everywhere mask == 1."""
    layer = frame.copy()
    layer[mask > 0] = color
    return cv2.addWeighted(frame, 1 - alpha, layer, alpha, 0)


def draw_arrow(frame, centerline):
    """Bold orange arrow following the path centerline (bottom -> top)."""
    pts = [(c, r) for r, c in centerline]   # centerline is [row, col]
    if len(pts) < 2:
        return
    pts = np.array(pts, dtype=np.int32)

    # Body: thick smooth line up the path.
    cv2.polylines(frame, [pts], False, ORANGE, thickness=16, lineType=cv2.LINE_AA)

    # Arrowhead at the top (last point), pointing in the path's direction.
    tip = pts[-1].astype(float)
    prev = pts[-2].astype(float)
    d = tip - prev
    n = np.linalg.norm(d)
    if n < 1e-3:
        return
    d /= n
    perp = np.array([-d[1], d[0]])          # perpendicular
    head_len, half_w = 38.0, 26.0
    apex = tip + d * head_len
    left = tip + perp * half_w
    right = tip - perp * half_w
    cv2.fillPoly(frame, [np.array([apex, left, right], dtype=np.int32)], ORANGE)


def draw_legend(frame, items):
    """Small chips at top: [(label, color), ...]."""
    x, y, pad, sw = 12, 12, 8, 22       # start, padding, swatch size
    for label, color in items:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        chip_w = sw + pad * 3 + tw
        chip_h = max(sw, th) + pad * 2
        # dark translucent background
        bg = frame.copy()
        cv2.rectangle(bg, (x, y), (x + chip_w, y + chip_h), (40, 40, 40), -1)
        cv2.addWeighted(bg, 0.5, frame, 0.5, 0, frame)
        # color swatch + label
        sy = y + (chip_h - sw) // 2
        cv2.rectangle(frame, (x + pad, sy), (x + pad + sw, sy + sw), color, -1)
        cv2.putText(frame, label, (x + pad * 2 + sw, y + chip_h - pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        x += chip_w + 10


def overlay(frame_bgr, result):
    """Draw the navigation overlay onto a copy of the frame."""
    h, w = result["height"], result["width"]
    frame = cv2.resize(frame_bgr, (w, h))

    walkable = clean(np.array(result["walkable_mask"], dtype=np.uint8))
    terrain = clean(np.array(result.get("terrain_mask", []), dtype=np.uint8)) \
        if result.get("terrain_mask") else np.zeros_like(walkable)
    road = clean(np.array(result.get("road_mask", []), dtype=np.uint8)) \
        if result.get("road_mask") else np.zeros_like(walkable)

    # Filled regions. Road + terrain are cyan; sidewalk green, painted last so
    # it wins any overlap.
    frame = fill_region(frame, terrain, CYAN)
    frame = fill_region(frame, road, CYAN)
    frame = fill_region(frame, walkable, GREEN)

    # Legend — only show classes actually present.
    items = []
    if walkable.any():
        items.append(("Sidewalk", GREEN))
    if road.any():
        items.append(("Road", CYAN))
    if terrain.any():
        items.append(("Terrain", CYAN))
    draw_legend(frame, items)

    # Direction arrow up the path.
    draw_arrow(frame, result.get("centerline", []))

    # Nearest obstacle: a small marker + warning (kept subtle).
    near = result.get("nearest_obstacle")
    if near:
        x = int(near["column_frac"] * w)
        cv2.line(frame, (x, h - 30), (x, h), RED, 4)
        cv2.putText(frame, f"OBSTACLE {near['distance_rel']:.2f}", (x - 70, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

    return frame


def perceive(frame_bgr):
    """Send one frame to the server; return the parsed result (or None on error)."""
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return None
    try:
        resp = requests.post(
            SERVER,
            files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("request failed:", e)
        return None


def main():
    if len(sys.argv) < 2:
        print('Usage: python live_view.py "path\\to\\walk.mp4" [frame_step]')
        return

    path = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Could not open video: {path}")
        return

    print("Working... the first frame is slowest. Press 'q' to quit.")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of video.")
            break
        if i % step == 0:
            result = perceive(frame)
            if result is not None:
                frame = overlay(frame, result)
            cv2.imshow("cane.ai — walking video (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
