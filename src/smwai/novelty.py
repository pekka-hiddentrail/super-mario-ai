"""Behaviour novelty: how different is a run from everything seen before, besides how far it got?

A run's fingerprint is the set of map cells (64 px squares of x/y) Mario visited plus the kinds of event it triggered. Two runs are
as far apart as their fingerprints overlap little (Jaccard distance). A genome's novelty is its mean distance to the k nearest
fingerprints in the archive and the rest of its generation. The archive keeps the most novel runs of each generation."""
import numpy as np

CELL = 64
EVENT_KINDS = ("powerup", "coin", "yoshi", "block", "pipe", "climb")


def cell_id(x, y):
    return (x // CELL) * 64 + min(63, max(0, y // CELL))


def fingerprint(cells, events=()):
    """cells: iterable of cell ids; events: [frame, kind, x, y] rows. Returns a frozenset of ints."""
    tokens = set(cells)
    for e in events:
        if e[1] in EVENT_KINDS:
            tokens.add(-1 - EVENT_KINDS.index(e[1]))
    return frozenset(tokens)


def distance(a, b):
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0


def knn_novelty(fp, others, k):
    if not others:
        return 0.0
    d = sorted(distance(fp, o) for o in others)
    return float(np.mean(d[:k]))


class NoveltyArchive:
    """Plain data (frozensets and numbers), so NEAT checkpoints can pickle the reporter that owns it."""

    def __init__(self, k=5, max_size=300, add_per_gen=3, min_add=0.05):
        self.k, self.max_size, self.add_per_gen, self.min_add = k, max_size, add_per_gen, min_add
        self.prints = []
        self.seen = set()                  # every cell any genome has ever reached: map coverage

    def score(self, prints):
        """prints: {genome id: fingerprint}. Returns {genome id: novelty} against the archive and the generation."""
        out = {}
        for gid, fp in prints.items():
            others = self.prints + [o for g, o in prints.items() if g != gid]
            out[gid] = knn_novelty(fp, others, self.k)
        return out

    def update(self, prints, scores):
        """Archive the most novel runs of this generation; returns how many map cells nobody had reached before."""
        for gid in sorted(scores, key=scores.get, reverse=True)[: self.add_per_gen]:
            if scores[gid] >= self.min_add:
                self.prints.append(prints[gid])
        del self.prints[: max(0, len(self.prints) - self.max_size)]
        before = len(self.seen)
        for fp in prints.values():
            self.seen.update(t for t in fp if t >= 0)
        return len(self.seen) - before


def blend(fitness, novelty, weight):
    """Selection score mixing min-max scaled fitness and novelty; the genome with the best raw fitness always ranks first."""
    ids = list(fitness)

    def scale(d):
        lo, hi = min(d.values()), max(d.values())
        return {i: (d[i] - lo) / (hi - lo) if hi > lo else 0.0 for i in ids}

    f, n = scale(fitness), scale(novelty)
    out = {i: (1 - weight) * f[i] + weight * n[i] for i in ids}
    out[max(ids, key=fitness.get)] += 1.0
    return out
