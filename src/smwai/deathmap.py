"""Memory across attempts: where did the best runs of recent generations die or get stuck, and where did they take off for the fatal fall?

The map is a histogram of end positions (32 px bins along the level, newer deaths weigh more). The network can read it as four
inputs: how much of the death mass is here, just ahead, further ahead, and how close the nearest known death spot ahead is.
A second table keeps the takeoff of the fatal jump (the last place with ground under Mario's feet, how long the jump button
was held, whether he was running) and gives four more inputs: how far the nearest bad takeoff is, how much of the takeoff mass
is just ahead, and how long jump was held / whether Y was held at it.
Death nodes (`death_nodes`) are the other way to use the same facts: each generation's best fatal takeoff leaves a node (a position and
a random nudge on one to all of the buttons) that pushes the network's button outputs near that spot, and fades with age.
Every attempt still starts at the level start; only the knowledge of earlier attempts is shared."""
import random

BIN = 32
NAMES = ("death_here", "death_near", "death_far", "death_dist")
TAKEOFF_NAMES = ("bad_dist", "bad_share", "bad_hold", "bad_run")
NODES = "nodes"                      # the death nodes inside DeathMap.export()
KEY = "takeoff"                      # the takeoff table inside DeathMap.export(); every other key is a bin number
TAKEOFF_DECAY = 0.5                  # a bad takeoff halves every generation and is gone after four (1, .5, .25, .125, .0625)
TAKEOFF_MIN = 0.07
MAX_HOLD = 16                        # steps of 4 frames; longer holds all read as 16


class DeathMap:
    """Plain lists and numbers, so NEAT checkpoints can pickle the reporter that owns it."""

    def __init__(self, decay=0.8, top_share=0.25):
        self.bins = {}
        self.takeoffs = {}           # bin -> [weight, hold sum, run sum] of the fatal takeoffs of the best runs
        self.decay, self.top_share = decay, top_share

    def update(self, ends, fitness, takeoffs=None):
        """ends: {genome id: [x, y, cleared]}; the not-cleared runs in the top share of fitness mark the hard spots.
        takeoffs: {genome id: [x, y, buttons mask, hold steps, run 0/1] or None}. Returns the takeoffs it kept (for the run's record)."""
        ranked = sorted(fitness, key=fitness.get, reverse=True)[: max(1, int(len(fitness) * self.top_share))]
        self.bins = {b: v * self.decay for b, v in self.bins.items() if v * self.decay > 0.01}
        self.takeoffs = {b: [w * TAKEOFF_DECAY, h * TAKEOFF_DECAY, r * TAKEOFF_DECAY] for b, (w, h, r) in self.takeoffs.items()
                         if w * TAKEOFF_DECAY > TAKEOFF_MIN}
        kept = []
        for gid in ranked:
            x, _y, cleared = ends[gid]
            if not cleared:
                self.bins[x // BIN] = self.bins.get(x // BIN, 0.0) + 1.0
                t = (takeoffs or {}).get(gid)
                if t:
                    w, h, r = self.takeoffs.get(t[0] // BIN, [0.0, 0.0, 0.0])
                    self.takeoffs[t[0] // BIN] = [w + 1.0, h + min(t[3], MAX_HOLD), r + t[4]]
                    kept.append(t + [round(fitness[gid], 1)])
        return kept

    def export(self):
        out = {str(b): round(v, 3) for b, v in self.bins.items()}
        if self.takeoffs:
            out[KEY] = {str(b): [round(w, 3), round(h, 3), round(r, 3)] for b, (w, h, r) in self.takeoffs.items()}
        return out


def features(bins, x):
    """Four ints in 0..16 for Mario at x; `bins` is DeathMap.export() (or None: no memory yet, all zeros)."""
    mass = {int(b): v for b, v in (bins or {}).items() if b.isdigit()}
    if not mass:
        return [0, 0, 0, 0]
    total = sum(mass.values())
    here = x // BIN

    def share(lo, hi):                       # fraction of all death mass in bins [here + lo, here + hi), saturating at 25%
        return min(16, round(16 * sum(v for b, v in mass.items() if here + lo <= b < here + hi) / (0.25 * total)))

    ahead = [b - here for b in mass if b >= here]
    dist = 16 - min(16, 2 * min(ahead)) if ahead else 0           # 16 = a known death spot right here, 0 = none within 8 bins (256 px)
    return [share(-1, 2), share(2, 4), share(4, 10), dist]


def takeoff_features(bins, x):
    """Four ints in 0..16 for Mario at x from the fatal-takeoff table: bad_dist (16 = a bad takeoff right here, 0 = none within
    256 px ahead), bad_share (how much of the takeoff mass is within the next 3 bins), bad_hold (mean jump hold steps at the nearest
    one ahead), bad_run (16 = Y was always held there, 0 = never)."""
    table = {int(b): v for b, v in ((bins or {}).get(KEY) or {}).items()}
    if not table:
        return [0, 0, 0, 0]
    here = x // BIN
    ahead = sorted(b for b in table if b >= here)
    if not ahead or ahead[0] - here >= 8:
        return [0, 0, 0, 0]
    total = sum(w for w, _, _ in table.values())
    near = sum(w for b, (w, _, _) in table.items() if here <= b < here + 3)
    w, h, r = table[ahead[0]]
    return [16 - 2 * (ahead[0] - here), min(16, round(16 * near / (0.25 * total))), min(16, round(h / w)), min(16, round(16 * r / w))]


# --- death nodes ---------------------------------------------------------------------------------------------
N_BUTTONS = 8                    # env.BUTTONS: A B X Y UP DOWN LEFT RIGHT
NODE_FADE_FAIL = 0.5             # per generation when no good attempt got past the node
NODE_FADE_WORKING = 0.85         # ... when a top attempt went through the node's area and past it
NODE_MIN_WEIGHT = 0.04           # gone below this: five generations of halving (1, 1/2, 1/4, 1/8, 1/16)
NODE_MAX_AGE = 20                # a node that keeps working still leaves after this many generations
NODE_REACH_X = (-32, 8)          # applies from 32 px before the takeoff to 8 px after it ...
NODE_REACH_Y = 48                # ... and within three tiles of its height, on the ground
NODE_CAP = 0.45                  # the summed nudge on one button (see NODE_LOGIT)
NODE_LOGIT = 12.0                # a nudge of 1.0 shifts the output's logit by this much: 0.45 -> +-5.4, which flips outputs between about 0.5% and 99.5%.
                                 # (Trained networks sit at 0.00 / 1.00 most of the time, so a nudge added to the 0..1 output itself never flipped anything.)


def random_nudge(rng=random):
    """A nudge on 1..8 random buttons (count from a bell curve around 4.5), each a random value in [-0.5, 0.5] with magnitude at least 0.1."""
    count = max(1, min(N_BUTTONS, round(rng.gauss(4.5, 1.5))))
    nudge = [0.0] * N_BUTTONS
    for b in rng.sample(range(N_BUTTONS), count):
        nudge[b] = rng.choice((-1, 1)) * rng.uniform(0.1, 0.5)
    return nudge


class NodeMemory:
    """Death nodes: [x, y, nudge per button, weight, age]. Plain lists, so NEAT checkpoints can pickle the reporter that owns it."""

    def __init__(self):
        self.nodes = []

    def update(self, takeoff, passed, rng=random):
        """takeoff: [x, y, ...] of the fatal takeoff of this generation's best attempt (or None). passed(x, y) -> True when a top attempt
        went through the area of that node and on past it: such a node fades slowly, any other fast."""
        keep = []
        for n in self.nodes:
            n[3] *= NODE_FADE_WORKING if passed(n[0], n[1]) else NODE_FADE_FAIL
            n[4] += 1
            if n[3] >= NODE_MIN_WEIGHT and n[4] < NODE_MAX_AGE:
                keep.append(n)
        if takeoff:
            keep.append([takeoff[0], takeoff[1], random_nudge(rng), 1.0, 0])
        self.nodes = keep

    def export(self):
        return [{"x": x, "y": y, "w": round(w, 4), "n": [round(v, 3) for v in nudge]} for x, y, nudge, w, _age in self.nodes]


def in_reach(nx, ny, x, y):
    """Does a node at (nx, ny) apply to Mario at (x, y)?"""
    return nx + NODE_REACH_X[0] <= x <= nx + NODE_REACH_X[1] and abs(y - ny) <= NODE_REACH_Y


def node_nudge(nodes, x, y):
    """Summed nudge per button (list of 8) from the exported nodes that apply at Mario's position, limited to +-NODE_CAP."""
    total = [0.0] * N_BUTTONS
    for n in nodes or ():
        if in_reach(n["x"], n["y"], x, y):
            for b in range(N_BUTTONS):
                total[b] += n["w"] * n["n"][b]
    return [max(-NODE_CAP, min(NODE_CAP, v)) for v in total]
