"""Memory across attempts: where did the best runs of recent generations die or get stuck?

The map is a histogram of end positions (32 px bins along the level, newer deaths weigh more). The network can read it as four
inputs: how much of the death mass is here, just ahead, further ahead, and how close the nearest known death spot ahead is.
Every attempt still starts at the level start; only the knowledge of earlier attempts is shared."""
BIN = 32
NAMES = ("death_here", "death_near", "death_far", "death_dist")


class DeathMap:
    """Plain lists and numbers, so NEAT checkpoints can pickle the reporter that owns it."""

    def __init__(self, decay=0.8, top_share=0.25):
        self.bins = {}
        self.decay, self.top_share = decay, top_share

    def update(self, ends, fitness):
        """ends: {genome id: [x, y, cleared]}; the not-cleared runs in the top share of fitness mark the hard spots."""
        ranked = sorted(fitness, key=fitness.get, reverse=True)[: max(1, int(len(fitness) * self.top_share))]
        self.bins = {b: v * self.decay for b, v in self.bins.items() if v * self.decay > 0.01}
        for gid in ranked:
            x, _y, cleared = ends[gid]
            if not cleared:
                self.bins[x // BIN] = self.bins.get(x // BIN, 0.0) + 1.0

    def export(self):
        return {str(b): round(v, 3) for b, v in self.bins.items()}


def features(bins, x):
    """Four ints in 0..16 for Mario at x; `bins` is DeathMap.export() (or None: no memory yet, all zeros)."""
    if not bins:
        return [0, 0, 0, 0]
    mass = {int(b): v for b, v in bins.items()}
    total = sum(mass.values())
    here = x // BIN

    def share(lo, hi):                       # fraction of all death mass in bins [here + lo, here + hi), saturating at 25%
        return min(16, round(16 * sum(v for b, v in mass.items() if here + lo <= b < here + hi) / (0.25 * total)))

    ahead = [b - here for b in mass if b >= here]
    dist = 16 - min(16, 2 * min(ahead)) if ahead else 0           # 16 = a known death spot right here, 0 = none within 8 bins (256 px)
    return [share(-1, 2), share(2, 4), share(4, 10), dist]
