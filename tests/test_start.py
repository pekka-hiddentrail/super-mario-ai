import random

import neat

from smwai import config, start

CFG = "configs/level1.toml"


def _population(tmp_path, hidden, links, size=5):
    cfg = config.load(CFG)
    cfg.neat.update(num_hidden=hidden, start_links=links, pop_size=size)
    path = tmp_path / "neat.cfg"
    path.write_text(cfg.neat_text())
    ncfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))
    pop = neat.Population(ncfg)
    for genome in pop.population.values():
        start.add_start_links(genome, ncfg.genome_config, links, random)
    return ncfg, pop


def test_exact_number_of_links_and_no_dead_end_hidden_nodes(tmp_path):
    for hidden, links in ((4, 20), (8, 40)):
        ncfg, pop = _population(tmp_path, hidden, links)
        outputs = set(ncfg.genome_config.output_keys)
        for genome in pop.population.values():
            assert len(genome.connections) == links
            assert len([k for k in genome.nodes if k not in outputs]) == hidden
            for h in (k for k in genome.nodes if k not in outputs):
                assert any(b == h for (_a, b) in genome.connections), "hidden node without an incoming link"
                assert any(a == h and b in outputs for (a, b) in genome.connections), "hidden node without a link to a button"


def test_start_links_overrides_the_fraction(tmp_path):
    cfg = config.load(CFG)
    assert "partial_direct" in cfg.neat_text()
    cfg.neat["start_links"] = 20
    assert "initial_connection = unconnected" in cfg.neat_text()
