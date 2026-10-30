"""A small, exact start: a genome with a given number of links (the hidden nodes come from `num_hidden`).

NEAT's own initial_connection options give a fraction of all possible links (5% of 700 inputs is hundreds of links, most of them noise).
With `start_links` every first-generation genome gets exactly that many links, chosen like this:
  - every hidden node gets one link from a random input and one link to a random button (so none starts as a dead end),
  - the rest are spread evenly over input -> hidden, hidden -> button and input -> button, at random places, no link twice.
"""


def add_start_links(genome, gcfg, total, rng):
    """Add `total` links (at least two per hidden node) to a genome that has none yet; returns how many it added."""
    inputs, outputs = list(gcfg.input_keys), list(gcfg.output_keys)
    hidden = [k for k in genome.nodes if k not in outputs]
    count = 0

    def add(source, target):
        nonlocal count
        if (source, target) in genome.connections:
            return
        innovation = gcfg.innovation_tracker.get_innovation_number(source, target, "initial_connection")
        link = genome.create_connection(gcfg, source, target, innovation)
        genome.connections[link.key] = link
        count += 1

    for h in hidden:
        add(rng.choice(inputs), h)
        add(h, rng.choice(outputs))
    kinds = [(inputs, hidden), (hidden, outputs), (inputs, outputs)] if hidden else [(inputs, outputs)]
    attempts = 0
    while count < total and attempts < 100 * max(total, 1):
        source, target = kinds[attempts % len(kinds)]
        attempts += 1
        add(rng.choice(source), rng.choice(target))
    return count
