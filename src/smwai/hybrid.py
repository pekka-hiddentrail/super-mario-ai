"""NEAT and PPO as one learner (a 'Lamarckian' hybrid).

Every generation NEAT evaluates the population as usual. Then the best few genomes are refined with PPO: the genome's own
network (its weights and biases, same structure) is turned into a PyTorch policy, played with some randomness on several
emulators, and improved by gradient steps on what that experience showed. If the refined genome plays better it is written
back into the genome, and NEAT carries on (selection, crossover, new connections and nodes) from the improved parents.

NEAT supplies structure and diverse solutions, PPO supplies fine adjustment of the weights from per-step feedback. There is one
population and one set of networks, not two models.

Policy: each of the 8 buttons is pressed with probability p = eps + (1 - 2 eps) * output, so the usual NEAT rule 'press if the
output is above 0.5' is the same as 'press if p is above 0.5' and PPO has a little randomness to learn from.
"""
import copy
import json
import time
from pathlib import Path

import gymnasium as gym
import neat
import numpy as np
import torch
from gymnasium import spaces

from .env import BUTTONS, SmwEnv
from .train import PoolEvaluator, eval_genome_full


class GenomeNet(torch.nn.Module):
    """A NEAT feed-forward genome as a differentiable network (same outputs as neat.nn.FeedForwardNetwork)."""

    def __init__(self, genome, config):
        super().__init__()
        gc = config.genome_config
        self.inputs, self.outputs = list(gc.input_keys), list(gc.output_keys)
        self.wmin, self.wmax = gc.weight_min_value, gc.weight_max_value
        self.bmin, self.bmax = gc.bias_min_value, gc.bias_max_value
        enabled = [k for k, c in genome.connections.items() if c.enabled]
        layers, _required = neat.graphs.feed_forward_layers(self.inputs, self.outputs, enabled)   # incl. bias-only nodes as layer 0
        self.col = {k: i for i, k in enumerate(self.inputs)}              # node key -> column of the value table
        self.conn_keys, self.node_keys, self.layers = [], [], []
        weights, biases, responses = [], [], []
        for layer in layers:
            nodes = sorted(layer)
            first = len(self.col)
            for n in nodes:
                self.col[n] = len(self.col)
            src, widx, dst = [], [], []
            for local, n in enumerate(nodes):
                for k in enabled:
                    if k[1] == n and k[0] in self.col:
                        src.append(self.col[k[0]])
                        widx.append(len(self.conn_keys))
                        dst.append(local)
                        self.conn_keys.append(k)
                        weights.append(genome.connections[k].weight)
            bidx = []
            for n in nodes:
                bidx.append(len(self.node_keys))
                self.node_keys.append(n)
                biases.append(genome.nodes[n].bias)
                responses.append(genome.nodes[n].response)
            self.layers.append((torch.tensor(src, dtype=torch.long), torch.tensor(widx, dtype=torch.long),
                                torch.tensor(dst, dtype=torch.long), torch.tensor(bidx, dtype=torch.long), first, len(nodes)))
        self.w = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.tensor(biases, dtype=torch.float32))
        self.register_buffer("resp", torch.tensor(responses, dtype=torch.float32))
        self.out_cols = [self.col.get(k, -1) for k in self.outputs]       # -1: output with no incoming connection

    def forward(self, x):
        """x: (batch, n_inputs) -> (batch, n_outputs) in (0, 1). A node with no incoming connection outputs sigmoid(bias), as in neat-python."""
        cols = x
        for src, widx, dst, bidx, _first, size in self.layers:
            contrib = cols[:, src] * self.w[widx]
            z = torch.zeros(x.shape[0], size).index_add(1, dst, contrib)
            z = self.b[bidx] + self.resp[bidx] * z
            cols = torch.cat([cols, torch.sigmoid(5.0 * z.clamp(-12.0, 12.0))], dim=1)     # neat-python's sigmoid
        zero = torch.zeros(x.shape[0], 1)
        padded = torch.cat([cols, zero], dim=1)
        return padded[:, [c if c >= 0 else padded.shape[1] - 1 for c in self.out_cols]]

    def probs(self, x, eps):
        return eps + (1.0 - 2.0 * eps) * self.forward(x)

    def write_back(self, genome):
        """Copy the (refined) weights and biases into a genome with the same structure."""
        with torch.no_grad():
            for k, w in zip(self.conn_keys, self.w.tolist()):
                genome.connections[k].weight = float(min(self.wmax, max(self.wmin, w)))
            for n, b in zip(self.node_keys, self.b.tolist()):
                genome.nodes[n].bias = float(min(self.bmax, max(self.bmin, b)))


class Critic(torch.nn.Module):
    """Estimates how much reward is still to come from an observation (the PPO baseline). Shared by all refinements."""

    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation. Arrays are (steps, envs); last_value is (envs,)."""
    steps = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last = delta + gamma * lam * not_done * last
        adv[t] = last
    return adv, adv + values


def ppo_update(net, critic, opt_pi, opt_v, batch, cfg, eps):
    """Clipped PPO updates on the genome's weights (net) and the shared critic."""
    obs, actions, old_logp, adv, ret = batch
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    for _ in range(cfg["epochs"]):
        order = torch.randperm(n)
        for start in range(0, n, cfg["minibatch"]):
            idx = order[start:start + cfg["minibatch"]]
            dist = torch.distributions.Bernoulli(net.probs(obs[idx], eps))
            logp = dist.log_prob(actions[idx]).sum(-1)
            ratio = torch.exp(logp - old_logp[idx])
            surrogate = torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * adv[idx])
            loss = -surrogate.mean() - cfg["ent_coef"] * dist.entropy().sum(-1).mean()
            opt_pi.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt_pi.step()
            vloss = 0.5 * ((critic(obs[idx]) - ret[idx]) ** 2).mean()
            opt_v.zero_grad()
            vloss.backward()
            opt_v.step()


class ButtonEnv(gym.Env):
    """SmwEnv with float observations and a scaled reward, for the PPO rollouts (buttons stay independent, like NEAT's)."""

    def __init__(self, env_cfg, scale):
        self.inner, self.scale = SmwEnv(env_cfg), scale
        self.action_space = spaces.MultiBinary(len(BUTTONS))
        self.observation_space = spaces.Box(-16.0, 16.0, self.inner.observation_space.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, term, trunc, info = self.inner.step(np.asarray(action, dtype=bool))
        return obs.astype(np.float32), reward * self.scale, term, trunc, info


def _env_factory(env_cfg, scale):
    def make():
        return ButtonEnv(env_cfg, scale)
    return make


def collect(vec, net, critic, obs, cfg):
    """Play cfg['steps'] steps on every emulator with the (stochastic) policy."""
    steps, n_envs = cfg["steps"], obs.shape[0]
    store = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
    for _ in range(steps):
        x = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = torch.distributions.Bernoulli(net.probs(x, cfg["explore_eps"]))
            act = dist.sample()
            store["logp"].append(dist.log_prob(act).sum(-1).numpy())
            store["val"].append(critic(x).numpy())
        store["obs"].append(x.numpy())
        store["act"].append(act.numpy())
        obs, rew, done, _infos = vec.step(act.numpy().astype(np.int8))
        store["rew"].append(rew.astype(np.float32))
        store["done"].append(done.astype(np.float32))
    with torch.no_grad():
        last_value = critic(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    arrays = {k: np.stack(v) for k, v in store.items()}
    adv, ret = gae(arrays["rew"], arrays["val"], arrays["done"], last_value, cfg["gamma"], cfg["lam"])
    flat = lambda a: torch.as_tensor(a.reshape(steps * n_envs, *a.shape[2:]), dtype=torch.float32)
    return (flat(arrays["obs"]), flat(arrays["act"]), flat(arrays["logp"]), flat(adv), flat(ret)), obs


class RefineSchedule:
    """When is PPO used? On a fixed schedule, or only while NEAT's best fitness has stopped improving (a hard spot such as a gap)."""

    def __init__(self, cfg):
        self.cfg, self.reference, self.stall = cfg, -np.inf, 0

    def due(self, generation, best_fitness):
        cfg = self.cfg
        if best_fitness > self.reference + cfg["min_improvement"]:
            self.reference, self.stall = best_fitness, 0
        else:
            self.stall += 1
        if generation < cfg["start_generation"]:
            return False
        if cfg["trigger"] == "stall":
            return self.stall >= cfg["stall_generations"]
        return (generation - cfg["start_generation"]) % cfg["every"] == 0


class HybridEvaluator(PoolEvaluator):
    """NEAT evaluation, then PPO refinement of the best genomes, written back when it helps."""

    def __init__(self, n_workers, stats, env_cfg, cfg, log_path=None):
        super().__init__(n_workers, stats)             # the NEAT worker pool first (fork before any emulator in this process)
        from stable_baselines3.common.vec_env import SubprocVecEnv
        torch.set_num_threads(2)
        self.cfg, self.env_cfg, self.log_path = cfg, env_cfg, log_path
        self.vec = SubprocVecEnv([_env_factory(env_cfg, cfg["reward_scale"]) for _ in range(cfg["n_envs"])], start_method="fork")
        self.obs = self.vec.reset()
        self.critic = Critic(self.vec.observation_space.shape[0])
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.generation = -1
        self.schedule = RefineSchedule(cfg)

    def evaluate(self, genomes, config):
        super().evaluate(genomes, config)
        self.generation += 1
        cfg = self.cfg
        if not self.schedule.due(self.generation, max(g.fitness for _, g in genomes)):
            return
        elites = sorted(genomes, key=lambda gg: gg[1].fitness, reverse=True)[: cfg["elites"]]
        report, t0 = [], time.time()
        for gid, genome in elites:
            before = genome.fitness
            refined = copy.deepcopy(genome)
            self._refine(refined, config)
            fitness, frames, behaviour = self.pool.apply_async(eval_genome_full, (refined, config)).get() if self.pool is not None \
                else eval_genome_full(refined, config)
            self.stats["frames"] += frames
            accepted = fitness > before + 1e-6
            if accepted:                                 # Lamarckian step: the learned weights are inherited
                for k, c in refined.connections.items():
                    genome.connections[k].weight = c.weight
                for k, nd in refined.nodes.items():
                    genome.nodes[k].bias = nd.bias
                genome.fitness = fitness
                self.stats.setdefault("behaviour", {})[gid] = behaviour
            report.append({"genome": gid, "before": round(before, 1), "after": round(fitness, 1), "accepted": accepted})
        line = ", ".join(f"{r['before']:.0f}->{r['after']:.0f}{'*' if r['accepted'] else ''}" for r in report)
        print(f"   hybrid: PPO refined {len(report)} genomes in {time.time() - t0:.0f}s: {line}   (* = kept)", flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"generation": self.generation, "hybrid": report}) + "\n")

    def _refine(self, genome, config):
        cfg = self.cfg
        net = GenomeNet(genome, config)
        opt_pi = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        for _ in range(cfg["iters"]):
            batch, self.obs = collect(self.vec, net, self.critic, self.obs, cfg)
            self.stats["frames"] += cfg["steps"] * cfg["n_envs"] * self.env_cfg.frame_skip
            ppo_update(net, self.critic, opt_pi, self.opt_v, batch, cfg, cfg["explore_eps"])
        net.write_back(genome)

    def close(self):
        super().close()
        self.vec.close()
