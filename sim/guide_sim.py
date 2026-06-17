"""
cane.ai — 2D guide-dog simulation (a beginner-friendly sandbox).

A top-down toy world where a virtual cart leads a person up a sidewalk like a
guide dog: it follows the corridor, yields to moving pedestrians, steers around
obstacles, and — at a street crossing — STOPS at the curb, waits for the
handler's command, and crosses only when it's actually safe.

────────────────────────────────────────────────────────────────────────────
THE GUIDE-DOG BEHAVIORS MODELLED HERE
  • Lead         — follow the sidewalk centerline.
  • Avoid        — keep the whole body (not just a point) clear of obstacles.
  • Yield        — slow / stop for moving pedestrians, then resume.
  • Stop at curb — a curb is a decision point; the cart halts and waits.
  • Cross safely — at a road crossing the HANDLER commands the cross (SPACE),
                   but the cart exercises "intelligent disobedience": it refuses
                   to go while a car is coming, and only crosses on a safe gap.

This is a STATE MACHINE (LEADING → WAIT_CROSS → CROSSING), which is how real
robot behaviour is structured (behaviour trees / state machines).

WHAT THIS TESTS — AND WHAT IT DOESN'T
  • It tests the DECISION logic, not perception. Because WE drew the world, the
    cart is simply TOLD where the sidewalk, obstacles, pedestrians, curb, and
    cars are — a stand-in for "perfect eyes." On the real device those facts
    come from the camera + LiDAR; the rules acting on them are the same.
  • The "recognise a real sidewalk/curb/car" half is tested on real footage —
    see live_view.py.
────────────────────────────────────────────────────────────────────────────

Setup (once):
    pip install opencv-python numpy

Run:
    python sim/guide_sim.py

Keys:  SPACE = command "cross" (at a crossing)    r = reset    q = quit
"""

import numpy as np
import cv2

# ── World layout (metres) ───────────────────────────────────────────────────
WORLD_W, WORLD_H = 6.0, 14.0
SIDEWALK_L, SIDEWALK_R = 1.5, 4.5   # a realistically wide sidewalk (room to pass)
SIDEWALK_CENTER = (SIDEWALK_L + SIDEWALK_R) / 2
ROAD_Y0, ROAD_Y1 = 5.5, 8.0        # the crossing: a road spanning this y-band
ROAD_CENTER = (ROAD_Y0 + ROAD_Y1) / 2
END_CURB = 13.0                    # final curb at the top of the far sidewalk
CROSSWALK_X0 = SIDEWALK_L - 0.2    # x-extent of the crosswalk (+ a margin)
CROSSWALK_X1 = SIDEWALK_R + 0.2

# ── Cart + control parameters ───────────────────────────────────────────────
CART_RADIUS = 0.28                 # body half-width; keep this clear of hazards
SAFETY_MARGIN = 0.18               # extra berth to leave around hazards/people
CRUISE_SPEED = 1.2                 # m/s walking pace
CROSS_SPEED = 1.6                  # m/s, cross a touch briskly
MAX_TURN = np.radians(140)         # max heading change rate (rad/s)
LOOKAHEAD = 2.0                    # m ahead to aim for on the centerline
STOP_DIST = 0.7                    # stop if the way forward is blocked within this
SLOW_DIST = 1.8                    # start slowing below this clearance
LIDAR_RANGE = 6.0
DT = 0.04
SAFE_GAP = 3.2                     # need this many seconds clear before crossing
CAR_SPAWN = 8.0                    # cars appear this far off-screen (sensed early)

# ── Rendering ───────────────────────────────────────────────────────────────
PX_PER_M = 50
IMG_W, IMG_H = int(WORLD_W * PX_PER_M), int(WORLD_H * PX_PER_M)


def w2p(x, y):
    """World metres -> image pixel (col, row). y points up in the world."""
    return int(x * PX_PER_M), int((WORLD_H - y) * PX_PER_M)


# ── Moving entities ─────────────────────────────────────────────────────────
class Cart:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = SIDEWALK_CENTER
        self.y = 0.4
        self.heading = np.pi / 2
        self.speed = 0.0


class Pedestrian:
    """A person walking down the sidewalk toward the cart (oncoming foot traffic).
    The cart yields/steers around them; they also sidestep around the cart, like
    real people — so they pass each other instead of deadlocking."""
    def __init__(self, x, y, vy, r=0.28):
        self.x, self.y, self.vy, self.r = x, y, vy, r

    def update(self, cart):
        dx, dy = self.x - cart.x, self.y - cart.y
        if dx * dx + dy * dy < (self.r + CART_RADIUS + 0.35) ** 2:
            side = np.sign(dx) if abs(dx) > 0.05 else 1.0
            self.x += side * 1.4 * DT          # step firmly aside
            self.y += self.vy * 0.4 * DT       # ease forward while squeezing past
        else:
            self.y += self.vy * DT
        self.x = min(max(self.x, SIDEWALK_L + self.r), SIDEWALK_R - self.r)


class PedTraffic:
    """Spawns oncoming pedestrians in the near and far sidewalk segments."""
    def __init__(self, rng):
        self.rng = rng
        self.peds = []
        self.timer = rng.uniform(1.0, 2.5)

    def update(self, cart):
        self.timer -= DT
        if self.timer <= 0:
            self.timer = self.rng.uniform(3.5, 6.0)
            speed = self.rng.uniform(0.5, 0.7)
            # Oncoming people keep to their lane (the cart's left), so the cart
            # passes them on the right with a clear gap instead of head-on.
            x = self.rng.uniform(SIDEWALK_L + 0.3, SIDEWALK_L + 0.6)
            top = (ROAD_Y0 - 0.5) if self.rng.random() < 0.5 else (END_CURB - 0.5)
            self.peds.append(Pedestrian(x, top, -speed))
        for p in self.peds:
            p.update(cart)
        # Remove once they walk off the bottom or reach the road.
        self.peds = [p for p in self.peds
                     if p.y > -0.6 and not (ROAD_Y0 <= p.y <= ROAD_Y1)]


class Car:
    HALF_X, HALF_Y = 0.45, 0.28
    def __init__(self, x, vx):
        self.x, self.vx, self.y = x, vx, ROAD_CENTER

    def update(self):
        self.x += self.vx * DT


class Traffic:
    """Spawns cars that drive across the road with random gaps."""
    def __init__(self, rng):
        self.rng = rng
        self.cars = []
        self.timer = rng.uniform(0.5, 1.5)

    def update(self):
        self.timer -= DT
        if self.timer <= 0:
            self.timer = self.rng.uniform(6.0, 9.0)        # gaps between cars
            speed = self.rng.uniform(2.0, 3.0)
            # Spawn well off-screen so the cart "senses" cars approaching with
            # enough warning to judge a safe gap (a real LiDAR sees them coming).
            if self.rng.random() < 0.5:
                self.cars.append(Car(-CAR_SPAWN, speed))
            else:
                self.cars.append(Car(WORLD_W + CAR_SPAWN, -speed))
        for c in self.cars:
            c.update()
        self.cars = [c for c in self.cars if -CAR_SPAWN - 1 < c.x < WORLD_W + CAR_SPAWN + 1]


def road_clear(cars):
    """Intelligent disobedience: only TRUE if no car is in or will reach the
    crosswalk within SAFE_GAP seconds."""
    for c in cars:
        if CROSSWALK_X0 <= c.x <= CROSSWALK_X1:
            return False
        if c.vx > 0 and c.x < CROSSWALK_X0 and (CROSSWALK_X0 - c.x) / c.vx < SAFE_GAP:
            return False
        if c.vx < 0 and c.x > CROSSWALK_X1 and (c.x - CROSSWALK_X1) / (-c.vx) < SAFE_GAP:
            return False
    return True


# ── Perception stand-in + control ───────────────────────────────────────────
def raycast(px, py, ang, obstacles, clearance=0.0, stop_line=1e9):
    """One 'LiDAR' ray: march until a hazard, return (distance, type).

    `clearance` inflates hazards by the cart's body radius so planning keeps the
    whole cart clear, not just its center. `stop_line` is the y of the curb the
    cart must not cross (a drop-off / the near edge of a road crossing).
    """
    step = 0.05
    d = 0.0
    while d < LIDAR_RANGE:
        d += step
        x = px + d * np.cos(ang)
        y = py + d * np.sin(ang)
        if y >= stop_line - clearance:
            return d, "curb"
        if x <= SIDEWALK_L + clearance or x >= SIDEWALK_R - clearance:
            return d, "road"
        for ox, oy, orr, otype in obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= (orr + clearance) ** 2:
                return d, otype
    return LIDAR_RANGE, None


def lead(cart, obstacles, stop_line):
    """Lead up the corridor: aim at the centerline, avoid by clearance, stop for
    curbs/blockages. Returns (target_heading, speed, state, reason_type)."""
    aim_x, aim_y = SIDEWALK_CENTER, cart.y + LOOKAHEAD
    desired = np.arctan2(aim_y - cart.y, aim_x - cart.x)
    clr = CART_RADIUS + SAFETY_MARGIN            # keep a clear berth, not just touch

    # Curb / stop-line straight ahead → stop (never roll off it).
    curb_d = LIDAR_RANGE
    for a in np.radians([-15, 0, 15]):
        d, t = raycast(cart.x, cart.y, desired + a, obstacles, clr, stop_line)
        if t == "curb":
            curb_d = min(curb_d, d)
    if curb_d < STOP_DIST:
        return cart.heading, 0.0, "STOP", "curb"

    # Avoid: pick the direction with the most clearance, preferring `desired`.
    best_ang, best_score, best_clear, best_type = desired, -1e9, 0.0, None
    for ang in desired + np.radians(np.arange(-70, 71, 5)):
        dist, htype = raycast(cart.x, cart.y, ang, obstacles, clr, stop_line)
        score = dist - 0.6 * abs(ang - desired)
        if score > best_score:
            best_ang, best_score, best_clear, best_type = ang, score, dist, htype

    if best_clear < STOP_DIST:
        return best_ang, 0.0, "STOP", best_type or "blocked"

    speed = CRUISE_SPEED
    if best_clear < SLOW_DIST:
        frac = (best_clear - STOP_DIST) / (SLOW_DIST - STOP_DIST)
        speed = max(0.2, CRUISE_SPEED * frac)
    return best_ang, speed, "GO", best_type


def step(cart, target_ang, target_speed):
    err = np.arctan2(np.sin(target_ang - cart.heading),
                     np.cos(target_ang - cart.heading))
    cart.heading += np.clip(err, -MAX_TURN * DT, MAX_TURN * DT)
    cart.speed = target_speed
    cart.x += cart.speed * DT * np.cos(cart.heading)
    cart.y += cart.speed * DT * np.sin(cart.heading)


def make_obstacles(rng):
    """Static obstacles staggered to alternating sides (always a weave path),
    placed in the near and far sidewalk segments — never on the road."""
    obs = []
    spans = [(2.6, ROAD_Y0 - 0.8), (ROAD_Y1 + 0.8, END_CURB - 1.2)]
    side = int(rng.integers(0, 2))
    k = 0
    for (y0, y1) in spans:
        for yc in np.linspace(y0 + 0.4, y1 - 0.4, 2):
            r = rng.uniform(0.25, 0.33)
            y = yc + rng.uniform(-0.25, 0.25)
            if (k + side) % 2 == 0:
                x = SIDEWALK_R - r - 0.05 - rng.uniform(0, 0.1)
            else:
                x = SIDEWALK_L + r + 0.05 + rng.uniform(0, 0.1)
            obs.append((x, y, r, "obstacle"))
            k += 1
    return obs


# ── The simulation (state machine) ──────────────────────────────────────────
REASON_TEXT = {
    "pedestrian": "yielding to pedestrian",
    "obstacle": "steering around obstacle",
    "road": "keeping off the road edge",
    "curb": "stopped at curb",
    "blocked": "path blocked — holding",
}


class Sim:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.cart = Cart()
        self.obstacles = make_obstacles(self.rng)
        self.pedtraffic = PedTraffic(self.rng)
        self.traffic = Traffic(self.rng)
        self.mode = "LEADING"
        self.cross_cmd = False
        self.state = "GO"
        self.reason = ""
        self.target = self.cart.heading

    def tick(self):
        self.traffic.update()
        self.pedtraffic.update(self.cart)
        peds = self.pedtraffic.peds
        dyn = [(o[0], o[1], o[2], "obstacle") for o in self.obstacles] + \
              [(p.x, p.y, p.r, "pedestrian") for p in peds]
        cart = self.cart
        crossed = cart.y >= ROAD_Y1

        if self.mode == "LEADING":
            stop_line = END_CURB if crossed else ROAD_Y0
            ang, spd, state, rtype = lead(cart, dyn, stop_line)
            # Emergency stop: never drive into contact, even from the side.
            gap = min([((cart.x - o0) ** 2 + (cart.y - o1) ** 2) ** 0.5 - o2 - CART_RADIUS
                       for o0, o1, o2, _ in dyn], default=9.0)
            if gap < 0.05 and spd > 0:
                spd, state, rtype = 0.0, "STOP", "pedestrian"
            step(cart, ang, spd)
            self.target, self.state = ang, state
            self.reason = REASON_TEXT.get(rtype, "") if state != "GO" else "leading"
            # Reached the crossing curb? (only near the road, not a random stop)
            if (not crossed) and state == "STOP" and cart.y > ROAD_Y0 - STOP_DIST - 0.5:
                self.mode = "WAIT_CROSS"

        elif self.mode == "WAIT_CROSS":
            cart.speed = 0.0
            self.state = "WAIT"
            self.target = cart.heading
            if self.cross_cmd and road_clear(self.traffic.cars):
                self.mode = "CROSSING"
            elif self.cross_cmd:
                self.reason = "REFUSING: car coming"          # intelligent disobedience
            else:
                self.reason = "at crossing — press SPACE to cross"

        elif self.mode == "CROSSING":
            # We only got here on a verified safe gap → commit and cross briskly,
            # straight to the far side (stopping in the road would be dangerous).
            aim = np.arctan2(1.2, SIDEWALK_CENTER - cart.x)      # up, re-centering
            step(cart, aim, CROSS_SPEED)
            self.target, self.state, self.reason = aim, "CROSS", "crossing"
            if cart.y >= ROAD_Y1 + 0.05:
                self.mode, self.cross_cmd = "LEADING", False


# ── Rendering ───────────────────────────────────────────────────────────────
def render(sim):
    cart = sim.cart
    img = np.full((IMG_H, IMG_W, 3), (70, 70, 70), np.uint8)        # generic ground

    # Sidewalk segments (near + far).
    for (y0, y1) in [(0, ROAD_Y0), (ROAD_Y1, WORLD_H)]:
        p0, p1 = w2p(SIDEWALK_L, y1), w2p(SIDEWALK_R, y0)
        cv2.rectangle(img, p0, p1, (130, 130, 130), -1)

    # Road band + crosswalk stripes.
    cv2.rectangle(img, w2p(0, ROAD_Y1), w2p(WORLD_W, ROAD_Y0), (50, 50, 55), -1)
    for sx in np.arange(SIDEWALK_L + 0.1, SIDEWALK_R, 0.45):
        cv2.rectangle(img, w2p(sx, ROAD_Y1 - 0.15), w2p(sx + 0.22, ROAD_Y0 + 0.15),
                      (220, 220, 220), -1)

    # Curbs (red): near edge of the crossing + final curb.
    cv2.line(img, w2p(SIDEWALK_L, ROAD_Y0), w2p(SIDEWALK_R, ROAD_Y0), (0, 0, 255), 3)
    cv2.line(img, w2p(0, END_CURB), w2p(WORLD_W, END_CURB), (0, 0, 255), 3)

    # Centerline (dashed) in the sidewalk segments.
    for yy in np.arange(0, WORLD_H, 0.5):
        if yy < ROAD_Y0 or yy > ROAD_Y1:
            cv2.line(img, w2p(SIDEWALK_CENTER, yy), w2p(SIDEWALK_CENTER, yy + 0.25),
                     (95, 95, 95), 1, cv2.LINE_AA)

    # Static obstacles (red), pedestrians (magenta), cars (steel blue).
    for ox, oy, orr, _ in sim.obstacles:
        cv2.circle(img, w2p(ox, oy), int(orr * PX_PER_M), (40, 40, 200), -1)
    for p in sim.pedtraffic.peds:
        cv2.circle(img, w2p(p.x, p.y), int(p.r * PX_PER_M), (200, 60, 200), -1)
    for c in sim.traffic.cars:
        if 0 <= c.x <= WORLD_W:                              # on-screen: draw the car
            cv2.rectangle(img, w2p(c.x - Car.HALF_X, c.y + Car.HALF_Y),
                          w2p(c.x + Car.HALF_X, c.y - Car.HALF_Y), (200, 140, 40), -1)
        elif (c.x < 0 and c.vx > 0) or (c.x > WORLD_W and c.vx < 0):
            # approaching but off-screen: a red "car incoming" arrow at the edge
            if c.x < 0:
                tip, b1, b2 = w2p(0.5, ROAD_CENTER), w2p(0.08, ROAD_CENTER + 0.35), w2p(0.08, ROAD_CENTER - 0.35)
            else:
                tip, b1, b2 = w2p(WORLD_W - 0.5, ROAD_CENTER), w2p(WORLD_W - 0.08, ROAD_CENTER + 0.35), w2p(WORLD_W - 0.08, ROAD_CENTER - 0.35)
            cv2.fillPoly(img, [np.array([tip, b1, b2])], (0, 0, 255))

    # LiDAR fan (faint, true distances) + chosen heading (yellow).
    for ang in cart.heading + np.radians(np.arange(-70, 71, 12)):
        d, _ = raycast(cart.x, cart.y, ang, [], 0.0,
                       1e9)  # show free space (entities omitted for clarity)
        cv2.line(img, w2p(cart.x, cart.y),
                 w2p(cart.x + d * np.cos(ang), cart.y + d * np.sin(ang)),
                 (0, 180, 180), 1, cv2.LINE_AA)

    # The cart triangle: green leading, orange crossing, red waiting/stopped.
    color = {"GO": (60, 200, 60), "CROSS": (0, 165, 255)}.get(sim.state, (60, 60, 230))
    tip = np.array([cart.x + 0.35 * np.cos(cart.heading),
                    cart.y + 0.35 * np.sin(cart.heading)])
    perp = np.array([-np.sin(cart.heading), np.cos(cart.heading)]) * 0.22
    base = np.array([cart.x, cart.y])
    cv2.fillPoly(img, [np.array([w2p(*tip), w2p(*(base + perp)), w2p(*(base - perp))])],
                 color)

    # HUD.
    mode_label = {"LEADING": "LEADING", "WAIT_CROSS": "AT CROSSING",
                  "CROSSING": "CROSSING"}[sim.mode]
    cv2.putText(img, f"{mode_label}   v={cart.speed:0.2f} m/s", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if sim.reason:
        warn = sim.reason.startswith("REFUSING") or "car" in sim.reason
        cv2.putText(img, sim.reason, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if warn else (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(img, "SPACE cross   r reset   q quit", (10, IMG_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def main():
    rng = np.random.default_rng()
    sim = Sim(rng)
    print("cane.ai guide sim — SPACE to cross, r reset, q quit.")
    while True:
        sim.tick()
        cv2.imshow("cane.ai — guide sim (SPACE cross, r reset, q quit)", render(sim))
        k = cv2.waitKey(int(DT * 1000)) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            sim.reset()
        if k == ord(" "):
            sim.cross_cmd = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
