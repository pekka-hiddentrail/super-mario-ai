"""Memory across attempts: where did the best runs of recent generations die or get stuck, and where did they take off for the fatal fall?

The map is a histogram of end positions (32 px bins along the level, newer deaths weigh more). The network can read it as four
inputs: how much of the death mass is here, just ahead, further ahead, and how close the nearest known death spot ahead is.
A second table keeps the takeoff of the fatal jump (the last place with ground under Mario's feet, how long the jump button
was held, whether he was running) and gives four more inputs: how far the nearest bad takeoff is, how much of the takeoff mass
is just ahead, and how long jump was held / whether Y was held at it.
Every attempt still starts at the level start; only the knowledge of earlier attempts is shared."""
BIN = 32
NAMES = ("death_here", "death_near", "death_far", "death_dist")
TAKEOFF_NAMES = ("bad_dist", "bad_share", "bad_hold", "bad_run")
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
    mass = {int(b): v for b, v in (bins or {}).items() if b != KEY}
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
