import neat
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smwai import config  # noqa: E402
from smwai.hybrid import Critic, GenomeNet, gae, ppo_update  # noqa: E402


def _population(tmp_path, n_in_cfg="configs/level1.toml"):
    cfg = config.load(n_in_cfg)
    path = tmp_path / "neat.cfg"
    path.write_text(cfg.neat_text())
    ncfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))
    return ncfg, neat.Population(ncfg)


def _grown_genome(tmp_path):
    ncfg, pop = _population(tmp_path)
    genome = next(iter(pop.population.values()))
    for _ in range(30):                                   # add nodes and connections, so there are hidden layers
        genome.mutate(ncfg.genome_config)
    return ncfg, genome


def test_genome_net_matches_neat_feed_forward(tmp_path):
    ncfg, genome = _grown_genome(tmp_path)
    assert any(k not in ncfg.genome_config.output_keys for k in genome.nodes), "test needs hidden nodes"
    ref = neat.nn.FeedForwardNetwork.create(genome, ncfg)
    net = GenomeNet(genome, ncfg)
    rng = np.random.default_rng(0)
    xs = rng.integers(-1, 3, size=(6, 169)).astype(np.float32)
    mine = net(torch.as_tensor(xs)).detach().numpy()
    for row, x in zip(mine, xs):
        assert np.allclose(row, ref.activate([float(v) for v in x]), atol=1e-4)


def test_write_back_roundtrip_keeps_the_network(tmp_path):
    ncfg, genome = _grown_genome(tmp_path)
    net = GenomeNet(genome, ncfg)
    with torch.no_grad():
        net.w.add_(0.37)
        net.b.sub_(0.11)
    net.write_back(genome)
    ref = neat.nn.FeedForwardNetwork.create(genome, ncfg)
    x = np.random.default_rng(1).integers(-1, 3, size=(1, 169)).astype(np.float32)
    assert np.allclose(net(torch.as_tensor(x)).detach().numpy()[0], ref.activate([float(v) for v in x[0]]), atol=1e-4)


def test_gae_matches_hand_calculation():
    rewards = np.array([[1.0], [0.0], [2.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    dones = np.zeros((3, 1), dtype=np.float32)
    adv, ret = gae(rewards, values, dones, np.zeros(1, dtype=np.float32), gamma=1.0, lam=1.0)
    assert np.allclose(ret[:, 0], [3.0, 2.0, 2.0])            # reward still to come from each step
    dones[0, 0] = 1.0                                         # an episode ends after step 0: later rewards do not count for it
    _, ret2 = gae(rewards, values, dones, np.zeros(1, dtype=np.float32), gamma=1.0, lam=1.0)
    assert np.isclose(ret2[0, 0], 1.0)


def test_ppo_update_raises_the_probability_of_the_rewarded_button(tmp_path):
    torch.manual_seed(0)
    ncfg, genome = _grown_genome(tmp_path)
    net = GenomeNet(genome, ncfg)
    critic = Critic(169)
    cfg = {**config.HYBRID_DEFAULTS, "epochs": 6, "minibatch": 128, "lr": 0.02}
    opt_pi = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    opt_v = torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"])
    eps = cfg["explore_eps"]
    obs = torch.as_tensor(np.random.default_rng(2).integers(-1, 3, size=(512, 169)).astype(np.float32))
    before = net.probs(obs, eps)[:, 0].mean().item()
    for _ in range(8):
        with torch.no_grad():
            dist = torch.distributions.Bernoulli(net.probs(obs, eps))
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)
        reward = act[:, 0]                                    # pressing button 0 is what pays
        adv = reward - reward.mean()
        ppo_update(net, critic, opt_pi, opt_v, (obs, act, logp, adv, reward), cfg, eps)
    assert net.probs(obs, eps)[:, 0].mean().item() > before + 0.02
