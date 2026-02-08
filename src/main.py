#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dlgpr_repro_final_complete.py
Reproducibility package for:
"A Dynamic Layered GA--PSO--RL Framework for Real-Time Game AI Under Compute Budgets"

UPDATED VERSION:
- Fixed NameError: generate_paper_figures is now defined.
- Includes RL Entropy Regularization.
- Includes GA Novelty Search support.
- Includes skeleton for Mixed-Integer parameter handling.
- Full integration of hard-budget accounting.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# ---------------------------- Optional Gym support ----------------------------
GYM_AVAILABLE = False
try:
    import gymnasium as gym  # type: ignore
    GYM_AVAILABLE = True
except Exception:
    try:
        import gym # type: ignore
        GYM_AVAILABLE = True
    except Exception:
        GYM_AVAILABLE = False


# ============================ Utilities =======================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)

def ema(prev: float, x: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * x

def median_mad(x: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return med, mad

def robust_z(x: float, window: List[float]) -> float:
    if len(window) == 0:
        return 0.0
    arr = np.asarray(window, dtype=float)
    med, mad = median_mad(arr)
    return (x - med) / (1.4826 * mad)

def safe_std(x: List[float]) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

def percentile(x: List[float], q: float) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.percentile(np.asarray(x, dtype=float), q))


# ============================ Environment =====================================

class SimpleDiscreteSpace:
    def __init__(self, n: int):
        self.n = n

class SimpleBoxSpace:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

class ToyPOMDP:
    """
    Lightweight POMDP-like episodic environment with:
    - discrete actions
    - continuous observation vector
    - stochastic reward
    - non-stationary opponent mode drift (to stress adaptation)
    """
    def __init__(self, obs_dim: int = 8, n_actions: int = 5, horizon: int = 64,
                 drift_prob: float = 0.02, noise_std: float = 0.15):
        self.observation_space = SimpleBoxSpace((obs_dim,))
        self.action_space = SimpleDiscreteSpace(n_actions)
        self.horizon = horizon
        self.drift_prob = drift_prob
        self.noise_std = noise_std

        self._rng = np.random.default_rng(0)
        self._t = 0
        self._mode = 0
        self._Wm = None  # shape (2, n_actions, obs_dim)
        self._init_params()

    def _init_params(self):
        obs_dim = self.observation_space.shape[0]
        nA = self.action_space.n
        rng = np.random.default_rng(123)
        W0 = rng.normal(0, 1.0, size=(nA, obs_dim))
        W1 = rng.normal(0, 1.0, size=(nA, obs_dim))
        W0 /= (np.linalg.norm(W0, axis=1, keepdims=True) + 1e-9)
        W1 /= (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-9)
        self._Wm = np.stack([W0, W1], axis=0)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._mode = int(self._rng.integers(0, 2))
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        d = self.observation_space.shape[0]
        phase = 2.0 * math.pi * (self._t / max(1, self.horizon))
        base = np.array([math.sin(phase), math.cos(phase)], dtype=float)
        ctx = self._rng.normal(0, 1.0, size=(d - 2,))
        obs = np.concatenate([base, ctx], axis=0).astype(float)
        return obs

    def step(self, action: int):
        if self._rng.random() < self.drift_prob:
            self._mode = 1 - self._mode

        obs = self._obs()
        w = self._Wm[self._mode, action]
        r = float(np.dot(w, obs) + self._rng.normal(0, self.noise_std))
        r = math.tanh(r)

        self._t += 1
        terminated = self._t >= self.horizon
        truncated = False
        info = {"mode": self._mode}
        return obs, r, terminated, truncated, info


def make_env(env_name: Optional[str], horizon: int, seed: int) -> Tuple[Any, int, int]:
    """
    Returns: (env, obs_dim, n_actions)
    If gymnasium available and env_name provided, uses gym; otherwise ToyPOMDP.
    """
    if env_name and GYM_AVAILABLE:
        try:
            env = gym.make(env_name)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=horizon)
            obs, _ = env.reset(seed=seed)
            obs_dim = int(np.asarray(obs).shape[0])
            if not hasattr(env.action_space, "n"):
                raise ValueError("Only discrete action spaces supported in this reference implementation.")
            n_actions = int(env.action_space.n)
            return env, obs_dim, n_actions
        except Exception as e:
            print(f"[WARN] Failed to load Gym env {env_name}: {e}. Falling back to ToyPOMDP.")

    env = ToyPOMDP(obs_dim=8, n_actions=5, horizon=horizon)
    env.reset(seed=seed)
    return env, env.observation_space.shape[0], env.action_space.n


# ============================ Policy ==========================================

@dataclass
class LinearSoftmaxPolicy:
    obs_dim: int
    n_actions: int

    @property
    def theta_dim(self) -> int:
        return self.n_actions * self.obs_dim + self.n_actions  # W + b

    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert theta.shape[0] == self.theta_dim
        W = theta[: self.n_actions * self.obs_dim].reshape(self.n_actions, self.obs_dim)
        b = theta[self.n_actions * self.obs_dim :].reshape(self.n_actions,)
        return W, b

    def act(self, theta: np.ndarray, obs: np.ndarray, rng: np.random.Generator,
            deterministic: bool) -> int:
        W, b = self.unpack(theta)
        logits = W @ obs + b
        pi = softmax(logits)
        if deterministic:
            return int(np.argmax(pi))
        return int(rng.choice(len(pi), p=pi))

    def logprob_and_grad(self, theta: np.ndarray, obs: np.ndarray, action: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Returns: (log pi(a|s), grad wrt theta, pi vector)
        Added pi vector return for entropy calculation.
        """
        W, b = self.unpack(theta)
        logits = W @ obs + b
        pi = softmax(logits)

        logp = float(np.log(pi[action] + 1e-12))

        # d log pi(a) / d logits = onehot(a) - pi
        dlogits = -pi
        dlogits[action] += 1.0  # (n_actions,)

        gW = np.outer(dlogits, obs)  # (n_actions, obs_dim)
        gb = dlogits
        grad = np.concatenate([gW.reshape(-1), gb], axis=0)
        return logp, grad, pi


# ============================ Evaluation ======================================

@dataclass
class EvalConfig:
    H: int = 64
    K_eval: int = 6
    gamma: float = 0.99
    deterministic_eval: bool = True

@dataclass
class TimeModel:
    rollout_ms_ema: float = 1.0
    train_rollout_ms_ema: float = 1.0
    alpha: float = 0.15
    safety_ms: float = 0.75

def _env_reset(env: Any, seed: int):
    if GYM_AVAILABLE and hasattr(env, "reset"):
        try:
            obs, info = env.reset(seed=seed)
            return obs, info
        except TypeError: # Old gym
             obs = env.reset(seed=seed)
             return obs, {}
    obs, info = env.reset(seed=seed)
    return obs, info

def rollout_return(env: Any, policy: LinearSoftmaxPolicy, theta: np.ndarray,
                   seed: int, H: int, gamma: float,
                   deterministic: bool) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)
    obs, _ = _env_reset(env, seed)
    obs = np.asarray(obs, dtype=float)

    G = 0.0
    steps = 0
    for t in range(H):
        a = policy.act(theta, obs, rng, deterministic=deterministic)
        step_out = env.step(a)
        # Handle gym 0.26+ vs old gym
        if len(step_out) == 5:
            obs2, r, terminated, truncated, _ = step_out
        else:
            obs2, r, done, _ = step_out
            terminated, truncated = done, False

        obs2 = np.asarray(obs2, dtype=float)
        G += (gamma ** t) * float(r)
        steps += 1
        obs = obs2
        if terminated or truncated:
            break
    return G, steps

def rollout_trajectory(env: Any, policy: LinearSoftmaxPolicy, theta: np.ndarray,
                       seed: int, H: int,
                       deterministic: bool) -> Tuple[List[np.ndarray], List[int], int]:
    rng = np.random.default_rng(seed)
    obs, _ = _env_reset(env, seed)
    obs = np.asarray(obs, dtype=float)

    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    steps = 0
    for _ in range(H):
        a = policy.act(theta, obs, rng, deterministic=deterministic)
        obs_list.append(obs.copy())
        act_list.append(int(a))
        step_out = env.step(a)
        if len(step_out) == 5:
            obs2, _, terminated, truncated, _ = step_out
        else:
            obs2, _, done, _ = step_out
            terminated, truncated = done, False
        obs = np.asarray(obs2, dtype=float)
        steps += 1
        if terminated or truncated:
            break
    return obs_list, act_list, steps

def evaluate_theta_strict(env: Any, policy: LinearSoftmaxPolicy, theta: np.ndarray,
                          eval_seeds: List[int], cfg: EvalConfig,
                          tmodel: TimeModel, deadline_ms: float) -> Optional[Tuple[float, float, int]]:
    """
    Strict evaluation with fixed K/H unless skipped.
    Returns None if insufficient time budget to complete K rollouts.
    """
    K = cfg.K_eval
    # conservative check (EMA-based)
    if now_ms() + K * tmodel.rollout_ms_ema + tmodel.safety_ms > deadline_ms:
        return None

    returns: List[float] = []
    steps_total = 0
    for s in eval_seeds[:K]:
        t0 = now_ms()
        G, steps = rollout_return(env, policy, theta, seed=s, H=cfg.H, gamma=cfg.gamma,
                                  deterministic=cfg.deterministic_eval)
        dt = now_ms() - t0
        tmodel.rollout_ms_ema = ema(tmodel.rollout_ms_ema, dt, tmodel.alpha)
        returns.append(G)
        steps_total += steps
        if now_ms() + (K - len(returns)) * tmodel.rollout_ms_ema + tmodel.safety_ms > deadline_ms:
            return None  # do not return partial; strict fixed-K

    meanJ = float(np.mean(returns)) if returns else 0.0
    se = float(np.std(returns, ddof=1) / math.sqrt(len(returns))) if len(returns) > 1 else 0.0
    return meanJ, se, steps_total

def behavioral_descriptor(env: Any, policy: LinearSoftmaxPolicy, theta: np.ndarray,
                          seed: int, H: int,
                          deadline_ms: float) -> Optional[np.ndarray]:
    """
    Descriptor: action-frequency vector over a short deterministic rollout.
    Returns None if deadline is exceeded before completion.
    """
    rng = np.random.default_rng(seed)
    obs, _ = _env_reset(env, seed)
    obs = np.asarray(obs, dtype=float)

    nA = policy.n_actions
    counts = np.zeros((nA,), dtype=float)
    for _ in range(H):
        if now_ms() > deadline_ms:
            return None
        a = policy.act(theta, obs, rng, deterministic=True)
        counts[a] += 1.0
        step_out = env.step(a)
        if len(step_out) == 5:
            obs2, _, terminated, truncated, _ = step_out
        else:
            obs2, _, done, _ = step_out
            terminated, truncated = done, False
        obs = np.asarray(obs2, dtype=float)
        if terminated or truncated:
            break
    return counts / (np.sum(counts) + 1e-12)


# ============================ Buffers =========================================

class EliteBuffer:
    def __init__(self, obs_dim: int, max_size: int = 5000):
        self.obs_dim = obs_dim
        self.max_size = max_size
        self.obs = np.zeros((max_size, obs_dim), dtype=float)
        self.act = np.zeros((max_size,), dtype=int)
        self.n = 0

    def add(self, obs_list: List[np.ndarray], act_list: List[int]) -> None:
        for o, a in zip(obs_list, act_list):
            idx = self.n % self.max_size
            self.obs[idx, :] = o
            self.act[idx] = int(a)
            self.n += 1

    def size(self) -> int:
        return min(self.n, self.max_size)

    def sample(self, rng: np.random.Generator, batch: int) -> Tuple[np.ndarray, np.ndarray]:
        m = self.size()
        if m == 0:
            return np.zeros((0, self.obs_dim)), np.zeros((0,), dtype=int)
        idx = rng.integers(0, m, size=(min(batch, m),))
        return self.obs[idx, :].copy(), self.act[idx].copy()


# ============================ Modules =========================================

@dataclass
class BudgetGuards:
    delta_min_ms: float = 5.0
    delta_max_ms: float = 30.0
    ema_alpha: float = 0.25
    eps: float = 1e-9

@dataclass
class ModuleSignals:
    hat_delta: float = 0.0
    hat_sigma: float = 0.0
    tilde_delta: float = 0.0
    barJ: float = 0.0
    last_tau: Optional[int] = None

@dataclass
class GAConfig:
    pop_size: int = 24
    offspring_per_step: int = 6
    elite_frac: float = 0.20
    p_crossover: float = 0.9
    p_mut: float = 0.35
    mut_std: float = 0.20
    tournament_k: int = 3
    # ADDED: Novelty weight for selection (Paper IV.F)
    novelty_weight: float = 0.1 

@dataclass
class PSOConfig:
    swarm_size: int = 18
    particles_per_step: int = 6
    chi: float = 0.729
    c1: float = 1.49445
    c2: float = 1.49445
    v_clip: float = 0.5

@dataclass
class RLConfig:
    # PPO-lite parameters
    lr: float = 0.02
    clip_eps: float = 0.2
    # ADDED: Entropy coefficient actually used in calculation
    ent_coef: float = 0.01 
    # BC distillation
    bc_lr: float = 0.01
    bc_batch: int = 64
    # rollout control
    train_rollouts_per_step: int = 1
    baseline_ema: float = 0.10

@dataclass
class ESConfig:
    sigma: float = 0.15
    lr: float = 0.20
    pop: int = 16

@dataclass
class EvaluatedCandidate:
    theta: np.ndarray
    J: float
    se: float
    novelty: float = 0.0  # ADDED for GA selection

def init_theta(rng: np.random.Generator, dim: int, scale: float = 0.5) -> np.ndarray:
    return rng.normal(0, scale, size=(dim,)).astype(float)

def project_theta(theta: np.ndarray) -> np.ndarray:
    """
    Project/Repair operator (Pi_Theta).
    Handles continuous constraints. 
    Skeleton for Mixed-Integer: Discrete parts would be rounded here.
    """
    # Continuous clipping
    clipped = np.clip(theta, -3.0, 3.0)
    
    # [PAPER-ALIGNMENT] If discrete params existed (theta^z), 
    # logic would be: theta[idx_discrete] = round(theta[idx_discrete])
    return clipped

class GAModule:
    def __init__(self, policy: LinearSoftmaxPolicy, cfg: GAConfig, rng: np.random.Generator):
        self.policy = policy
        self.cfg = cfg
        self.rng = rng
        self.P: List[np.ndarray] = [init_theta(rng, policy.theta_dim) for _ in range(cfg.pop_size)]
        self.last_fitness: List[float] = [-1e18 for _ in range(cfg.pop_size)]
        self.last_novelty: List[float] = [0.0 for _ in range(cfg.pop_size)] # ADDED

    def _tournament(self, fitness: List[float], novelty: List[float]) -> int:
        """
        Modified to use weighted sum of fitness and novelty if novelty > 0.
        """
        k = self.cfg.tournament_k
        idxs = self.rng.integers(0, len(self.P), size=(k,))
        best = int(idxs[0])
        
        # Calculate combined score
        # Normalize fitness roughly to 0-1 for combination if needed, 
        # but here we use simple weighted sum assuming roughly similar scales or relative rank
        def score(i):
            return fitness[i] + self.cfg.novelty_weight * novelty[i]

        best_score = score(best)
        for i in idxs[1:]:
            ii = int(i)
            s = score(ii)
            if s > best_score:
                best = ii
                best_score = s
        return best

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.cfg.p_crossover:
            return a.copy()
        mask = self.rng.random(size=a.shape) < 0.5
        child = np.where(mask, a, b)
        return child

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.cfg.p_mut:
            return x
        return x + self.rng.normal(0, self.cfg.mut_std, size=x.shape)

    def _compute_novelty(self, candidates: List[np.ndarray], descriptors: List[np.ndarray]) -> List[float]:
        """
        Computes novelty score (avg dist to k-NN in current population).
        """
        if len(descriptors) < 2:
            return [0.0] * len(candidates)
        
        novs = []
        d_stack = np.stack(descriptors)
        k = min(5, len(descriptors) - 1)
        
        for d in descriptors:
            dists = np.linalg.norm(d_stack - d, axis=1)
            dists.sort()
            # Mean distance to k nearest neighbors (excluding self at index 0)
            score = float(np.mean(dists[1:k+1]))
            novs.append(score)
        return novs

    def step(self, env: Any, eval_cfg: EvalConfig, eval_seeds: List[int],
             tmodel: TimeModel, timer_deadline_ms: float, 
             descriptor_seed: int, descriptor_H: int) -> Tuple[List[EvaluatedCandidate], float, float, int]:
        env_steps_used = 0

        # 1. Evaluate current population (fitness + descriptors)
        fitness = [-1e18 for _ in range(len(self.P))]
        se_list = [0.0 for _ in range(len(self.P))]
        descriptors: List[np.ndarray] = []

        # We need descriptors for the whole population to do novelty search properly
        # To save budget, we might cache, but for strictness we re-eval or assume static for one step.
        # Here we evaluate strictly.
        
        valid_indices = []

        for i, theta in enumerate(self.P):
            if now_ms() > timer_deadline_ms:
                break
            
            # Eval Fitness
            out = evaluate_theta_strict(env, self.policy, theta, eval_seeds, eval_cfg, tmodel, timer_deadline_ms)
            if out is None: break
            J, se, steps = out
            env_steps_used += int(steps)
            
            # Eval Descriptor (for novelty)
            # Use small budget for descriptor
            if now_ms() > timer_deadline_ms: break
            desc = behavioral_descriptor(env, self.policy, theta, descriptor_seed, descriptor_H, timer_deadline_ms)
            if desc is None: break
            
            fitness[i] = float(J)
            se_list[i] = float(se)
            descriptors.append(desc)
            valid_indices.append(i)

        if not valid_indices:
             return [], 0.0, 0.0, env_steps_used

        # Compute Novelty for those evaluated
        nov_scores = self._compute_novelty(self.P[:len(descriptors)], descriptors)
        
        # Update internal state (for injection replacement target)
        for idx, i in enumerate(valid_indices):
            self.last_fitness[i] = fitness[i]
            self.last_novelty[i] = nov_scores[idx]

        bestJ = float(max([fitness[i] for i in valid_indices]))
        best_se = float(se_list[valid_indices[int(np.argmax([fitness[i] for i in valid_indices]))]])

        # 2. Produce offspring
        children: List[np.ndarray] = []
        # Create a temp fitness/novelty list aligned with valid_indices for tournament
        valid_fit = [fitness[i] for i in valid_indices]
        valid_nov = nov_scores
        
        # If we didn't evaluate everyone, tournament only picks from valid ones
        # Temporary population mapping
        temp_pop = [self.P[i] for i in valid_indices]

        while len(children) < self.cfg.offspring_per_step and now_ms() <= timer_deadline_ms:
            # Need local tournament function that maps to temp_pop indices
            def local_tourn():
                k = self.cfg.tournament_k
                if len(temp_pop) < k: return 0
                idxs = self.rng.integers(0, len(temp_pop), size=(k,))
                best_idx = idxs[0]
                best_sc = valid_fit[best_idx] + self.cfg.novelty_weight * valid_nov[best_idx]
                for ii in idxs[1:]:
                    sc = valid_fit[ii] + self.cfg.novelty_weight * valid_nov[ii]
                    if sc > best_sc:
                        best_idx = ii
                        best_sc = sc
                return best_idx

            i = local_tourn()
            j = local_tourn()
            child = self._crossover(temp_pop[i], temp_pop[j])
            child = self._mutate(child)
            child = project_theta(child)
            children.append(child)

        # 3. Evaluate offspring
        evaluated: List[EvaluatedCandidate] = []
        child_fit: List[float] = []
        child_se: List[float] = []
        
        for child in children:
            if now_ms() > timer_deadline_ms: break
            out = evaluate_theta_strict(env, self.policy, child, eval_seeds, eval_cfg, tmodel, timer_deadline_ms)
            if out is None: break
            J, se, steps = out
            env_steps_used += int(steps)
            
            # Offspring novelty is unknown until next gen, set to 0 or compute?
            # For efficiency we don't compute novelty for offspring immediately 
            # unless we want to use it for survival. Standard GA: survival is usually fitness based or random.
            
            cand = EvaluatedCandidate(theta=child.copy(), J=float(J), se=float(se), novelty=0.0)
            evaluated.append(cand)
            child_fit.append(float(J))
            child_se.append(float(se))

        # 4. Elitist replacement (using Fitness primarily)
        # Combine valid parents + evaluated children
        elite_n = max(1, int(self.cfg.elite_frac * self.cfg.pop_size))
        
        # Sort parents by fitness
        parent_indices = np.argsort(valid_fit)[::-1] # indices into temp_pop
        elites = [temp_pop[i].copy() for i in parent_indices[:elite_n]]

        combined = elites[:]
        if len(child_fit) > 0:
            new_sorted = list(np.argsort(np.asarray(child_fit)))[::-1]
            for idx in new_sorted:
                combined.append(evaluated[idx].theta.copy())
                if len(combined) >= self.cfg.pop_size: break
        
        while len(combined) < self.cfg.pop_size:
            combined.append(self.P[int(self.rng.integers(0, len(self.P)))].copy())

        self.P = combined[: self.cfg.pop_size]

        # Best of step logic (among new children)
        if len(child_fit) > 0:
            mx = float(max(child_fit))
            if mx > bestJ:
                bestJ = mx
                best_se = float(child_se[int(np.argmax(child_fit))])

        return evaluated, float(bestJ), float(best_se), int(env_steps_used)


class PSOModule:
    def __init__(self, policy: LinearSoftmaxPolicy, cfg: PSOConfig, rng: np.random.Generator):
        self.policy = policy
        self.cfg = cfg
        self.rng = rng
        dim = policy.theta_dim
        self.swarm = [init_theta(rng, dim) for _ in range(cfg.swarm_size)]
        self.vel = [rng.normal(0, 0.1, size=(dim,)).astype(float) for _ in range(cfg.swarm_size)]
        self.pbest = [x.copy() for x in self.swarm]
        self.pbestJ = [-1e9 for _ in range(cfg.swarm_size)]
        self.gbest = self.swarm[0].copy()
        self.gbestJ = -1e9

    def step(self, env: Any, eval_cfg: EvalConfig, eval_seeds: List[int],
             tmodel: TimeModel, timer_deadline_ms: float) -> Tuple[List[EvaluatedCandidate], float, float, int]:
        env_steps_used = 0
        evaluated: List[EvaluatedCandidate] = []
        bestJ = -1e18
        best_se = 0.0

        idxs = self.rng.integers(0, self.cfg.swarm_size, size=(self.cfg.particles_per_step,))
        for idx in idxs:
            if now_ms() > timer_deadline_ms:
                break
            i = int(idx)
            x = self.swarm[i]
            v = self.vel[i]

            r1 = self.rng.random(size=x.shape)
            r2 = self.rng.random(size=x.shape)
            v = self.cfg.chi * (v
                                + self.cfg.c1 * r1 * (self.pbest[i] - x)
                                + self.cfg.c2 * r2 * (self.gbest - x))
            v = np.clip(v, -self.cfg.v_clip, self.cfg.v_clip)
            x2 = project_theta(x + v)

            out = evaluate_theta_strict(env, self.policy, x2, eval_seeds, eval_cfg, tmodel, timer_deadline_ms)
            if out is None:
                break
            J, se, steps = out
            env_steps_used += int(steps)

            self.swarm[i] = x2
            self.vel[i] = v

            if J > self.pbestJ[i]:
                self.pbestJ[i] = float(J)
                self.pbest[i] = x2.copy()
            if J > self.gbestJ:
                self.gbestJ = float(J)
                self.gbest = x2.copy()

            evaluated.append(EvaluatedCandidate(theta=x2.copy(), J=float(J), se=float(se)))
            if J > bestJ:
                bestJ = float(J)
                best_se = float(se)

        if bestJ < -1e17:
            return [], 0.0, 0.0, int(env_steps_used)
        return evaluated, float(bestJ), float(best_se), int(env_steps_used)


class RLModule:
    """
    PPO-lite (clipped surrogate) + BC distillation (optional).
    UPDATED: Includes entropy regularization.
    """
    def __init__(self, policy: LinearSoftmaxPolicy, cfg: RLConfig, rng: np.random.Generator, elite_buf: EliteBuffer):
        self.policy = policy
        self.cfg = cfg
        self.rng = rng
        self.phi = init_theta(rng, policy.theta_dim, scale=0.4)
        self.baseline = 0.0
        self.loss_ema = 0.0
        self.last_loss: Optional[float] = None
        self.elite_buf = elite_buf

    def _bc_step(self, deadline_ms: float) -> None:
        if self.elite_buf.size() == 0:
            return
        if now_ms() > deadline_ms:
            return
        X, A = self.elite_buf.sample(self.rng, self.cfg.bc_batch)
        if X.shape[0] == 0:
            return
        grad = np.zeros((self.policy.theta_dim,), dtype=float)
        for obs, a in zip(X, A):
            if now_ms() > deadline_ms:
                return
            _, g, _ = self.policy.logprob_and_grad(self.phi, obs, int(a))
            # BC loss = -logpi(a|s) -> grad = -grad_logp
            grad += (-g)
        grad /= float(X.shape[0])
        self.phi = project_theta(self.phi - self.cfg.bc_lr * grad)

    def _train_rollout(self, env: Any, seed: int, H: int, gamma: float,
                       deadline_ms: float, tmodel: TimeModel) -> Optional[Tuple[List[np.ndarray], List[int], List[float], List[float], int]]:
        # conservative check using EMA
        if now_ms() + tmodel.train_rollout_ms_ema + tmodel.safety_ms > deadline_ms:
            return None

        t0 = now_ms()
        rng = np.random.default_rng(seed)
        obs, _ = _env_reset(env, seed)
        obs = np.asarray(obs, dtype=float)

        obs_list: List[np.ndarray] = []
        act_list: List[int] = []
        rew_list: List[float] = []
        old_logp_list: List[float] = []
        steps = 0

        for _ in range(H):
            if now_ms() > deadline_ms:
                return None  # strict
            a = self.policy.act(self.phi, obs, rng, deterministic=False)
            logp, _, _ = self.policy.logprob_and_grad(self.phi, obs, int(a))

            step_out = env.step(int(a))
            if len(step_out) == 5:
                obs2, r, terminated, truncated, _ = step_out
            else:
                obs2, r, done, _ = step_out
                terminated, truncated = done, False

            obs_list.append(obs.copy())
            act_list.append(int(a))
            rew_list.append(float(r))
            old_logp_list.append(float(logp))

            obs = np.asarray(obs2, dtype=float)
            steps += 1
            if terminated or truncated:
                break

        dt = now_ms() - t0
        tmodel.train_rollout_ms_ema = ema(tmodel.train_rollout_ms_ema, dt, tmodel.alpha)
        return obs_list, act_list, rew_list, old_logp_list, steps

    def step(self, env: Any, eval_cfg: EvalConfig, train_seed: int,
             eval_seeds: List[int], tmodel: TimeModel,
             timer_deadline_ms: float) -> Tuple[List[EvaluatedCandidate], float, float, int, float]:
        env_steps_used = 0

        # BC distillation first
        self._bc_step(timer_deadline_ms)

        roll = self._train_rollout(env, train_seed, eval_cfg.H, eval_cfg.gamma, timer_deadline_ms, tmodel)
        if roll is None:
            return [], 0.0, 0.0, int(env_steps_used), float(self.loss_ema)

        obs_list, act_list, rew_list, old_logp_list, steps = roll
        env_steps_used += int(steps)

        # returns
        Gt: List[float] = []
        G = 0.0
        for r in reversed(rew_list):
            G = r + eval_cfg.gamma * G
            Gt.append(G)
        Gt = list(reversed(Gt))

        meanG = float(np.mean(Gt)) if len(Gt) else 0.0
        self.baseline = (1.0 - self.cfg.baseline_ema) * self.baseline + self.cfg.baseline_ema * meanG

        adv = np.asarray([g - self.baseline for g in Gt], dtype=float)

        # PPO-lite clipped surrogate + Entropy
        grad = np.zeros((self.policy.theta_dim,), dtype=float)
        loss_terms: List[float] = []

        for obs, a, adv_t, old_logp in zip(obs_list, act_list, adv, old_logp_list):
            if now_ms() > timer_deadline_ms:
                return [], 0.0, 0.0, int(env_steps_used), float(self.loss_ema)

            logp, g, pi_vec = self.policy.logprob_and_grad(self.phi, obs, int(a))
            ratio = float(np.exp(logp - old_logp))
            clipped = float(np.clip(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps))

            # PPO Loss
            s1 = ratio * float(adv_t)
            s2 = clipped * float(adv_t)
            s = min(s1, s2)
            
            # Entropy: H(pi) = -sum(pi * log(pi))
            entropy = -np.sum(pi_vec * np.log(pi_vec + 1e-12))
            
            # Maximize: s + ent_coef * entropy
            # Minimize: -s - ent_coef * entropy
            loss_t = -s - self.cfg.ent_coef * entropy
            loss_terms.append(loss_t)

            # Gradient wrt PPO part
            use_ratio = (s1 <= s2)
            if use_ratio:
                grad += -(float(adv_t) * ratio) * g 
            elif abs(clipped - ratio) < 1e-12:
                grad += -(float(adv_t) * ratio) * g
            else:
                grad += 0.0
            
            # Gradient wrt Entropy part: grad(H) = - (1 + log(pi)) * grad_pi 
            # where grad_pi = pi * (one_hot - pi) * obs
            # Simplified: dH/dlogits = pi * (log(pi) + 1) - pi * sum(pi(log(pi)+1)) ... 
            # Analytical grad of entropy wrt logits is: -pi * (log(pi) + 1) + pi * sum(pi(log(pi)+1))
            # Standard result for Softmax Policy: grad_theta H = sum( dH/dlogits * dlogits/dtheta )
            # We approximate entropy grad with simple heuristic or analytical if precise.
            # Analytical: grad_H = - pi * (log_pi + 1) [wrt logits]. 
            # Let's do it properly via autodiff logic emulation:
            # dH/dlogit_j = pi_j * ( -log_pi_j - 1 - sum_k(pi_k * (-log_pi_k - 1)) )
            #             = -pi_j * log_pi_j - pi_j + pi_j * (Entropy + 1)
            #             = pi_j * (Entropy - log_pi_j)
            
            dH_dlogits = pi_vec * (entropy - np.log(pi_vec + 1e-12))
            # grad_theta = dH_dlogits outer obs
            g_ent_W = np.outer(dH_dlogits, obs)
            g_ent_b = dH_dlogits
            grad_ent = np.concatenate([g_ent_W.reshape(-1), g_ent_b], axis=0)
            
            # Add negative gradient (since we descend on loss) of negative entropy term
            # Loss += - ent_coef * H
            # Grad += - ent_coef * grad_H
            grad += -self.cfg.ent_coef * grad_ent

        if len(obs_list) > 0:
            grad /= float(len(obs_list))
        loss = float(np.mean(loss_terms)) if len(loss_terms) else 0.0

        self.phi = project_theta(self.phi - self.cfg.lr * grad)

        # learning progress
        if self.last_loss is None:
            d = 0.0
        else:
            denom = max(abs(self.last_loss), 1e-9)
            d = (self.last_loss - loss) / denom
        self.last_loss = float(loss)
        self.loss_ema = ema(self.loss_ema, d, alpha=0.25)

        # eval
        out = evaluate_theta_strict(env, self.policy, self.phi, eval_seeds, eval_cfg, tmodel, timer_deadline_ms)
        if out is None:
            return [], 0.0, 0.0, int(env_steps_used), float(self.loss_ema)
        J, se, ev_steps = out
        env_steps_used += int(ev_steps)

        cand = EvaluatedCandidate(theta=self.phi.copy(), J=float(J), se=float(se))
        return [cand], float(J), float(se), int(env_steps_used), float(self.loss_ema)


class ESBaseline:
    def __init__(self, policy: LinearSoftmaxPolicy, cfg: ESConfig, rng: np.random.Generator):
        self.policy = policy
        self.cfg = cfg
        self.rng = rng
        self.mu = init_theta(rng, policy.theta_dim, scale=0.4)

    def step(self, env: Any, eval_cfg: EvalConfig, eval_seeds: List[int],
             tmodel: TimeModel, timer_deadline_ms: float) -> Tuple[List[EvaluatedCandidate], float, float, int]:
        env_steps_used = 0
        dim = self.policy.theta_dim
        eps_list: List[np.ndarray] = []
        J_list: List[float] = []
        evals: List[EvaluatedCandidate] = []

        bestJ = -1e18
        bestSE = 0.0

        for _ in range(self.cfg.pop):
            if now_ms() > timer_deadline_ms:
                break
            eps = self.rng.normal(0, 1.0, size=(dim,))
            theta = project_theta(self.mu + self.cfg.sigma * eps)

            out = evaluate_theta_strict(env, self.policy, theta, eval_seeds, eval_cfg, tmodel, timer_deadline_ms)
            if out is None:
                break
            J, se, steps = out
            env_steps_used += int(steps)

            eps_list.append(eps)
            J_list.append(float(J))
            evals.append(EvaluatedCandidate(theta=theta.copy(), J=float(J), se=float(se)))

            if J > bestJ:
                bestJ = float(J)
                bestSE = float(se)

        if len(J_list) == 0:
            return [], 0.0, 0.0, int(env_steps_used)

        J_arr = np.asarray(J_list, dtype=float)
        A = (J_arr - np.mean(J_arr)) / (np.std(J_arr) + 1e-12)

        grad = np.zeros((dim,), dtype=float)
        for a, eps in zip(A, eps_list):
            grad += float(a) * eps
        grad /= float(len(eps_list))

        self.mu = project_theta(self.mu + self.cfg.lr * (grad / (self.cfg.sigma + 1e-12)))
        return evals, float(bestJ), float(bestSE), int(env_steps_used)


# ============================ Scheduler / Runners ===============================

@dataclass
class SchedulerConfig:
    W: int = 30
    beta: float = 0.5
    lambda_D: float = 0.7
    lambda_L: float = 0.7
    lambda_sigma: float = 0.3
    n_min: int = 1
    fixed_split: Optional[Tuple[float, float, float]] = None

@dataclass
class DLGPRConfig:
    B_tau_ms: float = 120.0
    guards: BudgetGuards = field(default_factory=BudgetGuards)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    es: ESConfig = field(default_factory=ESConfig)
    diversity_charge_ms: float = 6.0
    descriptor_seed: int = 777
    descriptor_H: int = 24
    p_inj: float = 0.10

class DLGPRRunner:
    def __init__(self, env: Any, policy: LinearSoftmaxPolicy, cfg: DLGPRConfig, seed: int,
                 enable_ga: bool = True, enable_pso: bool = True, enable_rl: bool = True,
                 enable_diversity: bool = True, method_name: str = "DLGPR"):
        self.env = env
        self.policy = policy
        self.cfg = cfg
        self.seed = seed
        self.method_name = method_name

        self.rng = np.random.default_rng(seed)
        self.tmodel = TimeModel()

        self.enable_ga = enable_ga
        self.enable_pso = enable_pso
        self.enable_rl = enable_rl
        self.enable_diversity = enable_diversity

        self.ga = GAModule(policy, cfg.ga, self.rng) if enable_ga else None
        self.pso = PSOModule(policy, cfg.pso, self.rng) if enable_pso else None

        self.elite_buf = EliteBuffer(obs_dim=policy.obs_dim, max_size=5000)
        self.rl = RLModule(policy, cfg.rl, self.rng, self.elite_buf) if enable_rl else None

        self.signals: Dict[str, ModuleSignals] = {"GA": ModuleSignals(), "PSO": ModuleSignals(), "RL": ModuleSignals()}
        self.Nm = {"GA": 0, "PSO": 0, "RL": 0}
        self.Ntot = 0

        self.window_tilde = {"GA": [], "PSO": [], "RL": []}
        self.window_dD: List[float] = []
        self.window_dL: List[float] = []

        self.D_pop = 0.0
        self.prev_D_pop: Optional[float] = None
        self.L_rl = 0.0
        self.prev_L_rl: Optional[float] = None

    def _compute_diversity(self, deadline_ms: float) -> float:
        if not self.enable_ga or self.ga is None or not self.enable_diversity:
            return self.D_pop
        P = self.ga.P
        desc: List[np.ndarray] = []
        for th in P:
            d = behavioral_descriptor(self.env, self.policy, th,
                                     seed=self.cfg.descriptor_seed, H=self.cfg.descriptor_H,
                                     deadline_ms=deadline_ms)
            if d is None:
                break
            desc.append(d)
        m = len(desc)
        if m < 2:
            return self.D_pop
        pairs = min(40, m * (m - 1) // 2)
        dsum = 0.0
        for _ in range(pairs):
            if now_ms() > deadline_ms:
                break
            i = int(self.rng.integers(0, m))
            j = int(self.rng.integers(0, m))
            if i == j:
                continue
            dsum += float(np.linalg.norm(desc[i] - desc[j]))
        return dsum / max(1, pairs)

    def _dD_last(self) -> float:
        if self.prev_D_pop is None:
            return 0.0
        return float(self.D_pop - self.prev_D_pop)

    def _dL_last(self) -> float:
        if self.prev_L_rl is None:
            return 0.0
        return float(self.L_rl - self.prev_L_rl)

    def _update_module_signal_if_eval(self, m: str, tau: int, barJ: float, se: float, charged_ms: float) -> None:
        sig = self.signals[m]
        if sig.last_tau is None:
            dJ = 0.0
            sig.hat_delta = 0.0
            sig.hat_sigma = 0.0
        else:
            dJ = float(barJ - sig.barJ)
            rate = dJ / max(charged_ms, self.cfg.guards.eps)
            sig.hat_delta = ema(sig.hat_delta, rate, self.cfg.guards.ema_alpha)
            sig.hat_sigma = ema(sig.hat_sigma, (float(se) / max(charged_ms, self.cfg.guards.eps)), self.cfg.guards.ema_alpha)

        sig.barJ = float(barJ)
        sig.tilde_delta = sig.hat_delta - self.cfg.sched.lambda_sigma * sig.hat_sigma
        sig.last_tau = int(tau)

        self.window_tilde[m].append(float(sig.tilde_delta))
        if len(self.window_tilde[m]) > self.cfg.sched.W:
            self.window_tilde[m] = self.window_tilde[m][-self.cfg.sched.W:]

    def _module_index(self, m: str) -> float:
        if (m == "GA" and not self.enable_ga) or (m == "PSO" and not self.enable_pso) or (m == "RL" and not self.enable_rl):
            return -1e18

        z_rate = robust_z(self.signals[m].tilde_delta, self.window_tilde[m])
        ucb = self.cfg.sched.beta * math.sqrt(math.log(1.0 + self.Ntot) / (1.0 + self.Nm[m]))
        idx = z_rate + ucb

        if m == "GA" and self.enable_diversity:
            idx += self.cfg.sched.lambda_D * robust_z(self._dD_last(), self.window_dD)
        if m == "RL":
            idx += self.cfg.sched.lambda_L * robust_z(self._dL_last(), self.window_dL)
        return float(idx)

    def run(self, intervals: int, eval_cfg: EvalConfig,
            eval_seed_base: int, train_seed_base: int,
            out_dir: str) -> Dict[str, float]:
        ensure_dir(out_dir)
        logs_dir = os.path.join(out_dir, "logs")
        figs_dir = os.path.join(out_dir, "figs")
        tables_dir = os.path.join(out_dir, "tables")
        ensure_dir(logs_dir); ensure_dir(figs_dir); ensure_dir(tables_dir)

        rows: List[Dict[str, Any]] = []
        per_interval_score: List[float] = []
        per_interval_envsteps_online: List[int] = []
        per_interval_compute_ms: List[float] = []

        for tau in range(1, intervals + 1):
            div_reserved = float(self.cfg.diversity_charge_ms) if (self.enable_ga and self.enable_diversity) else 0.0
            B_rem = float(self.cfg.B_tau_ms) - div_reserved
            if B_rem < 0.0:
                B_rem = 0.0

            n = {"GA": 0, "PSO": 0, "RL": 0}
            csum = {"GA": 0.0, "PSO": 0.0, "RL": 0.0}
            C_tau: List[Tuple[np.ndarray, float, float, str]] = []

            self.prev_D_pop = self.D_pop
            self.prev_L_rl = self.L_rl

            dL = self._dL_last()
            self.window_dL.append(float(dL))
            if len(self.window_dL) > self.cfg.sched.W:
                self.window_dL = self.window_dL[-self.cfg.sched.W:]

            env_steps_online = 0
            t_comp0 = now_ms()

            while B_rem > 0.0:
                if B_rem < self.cfg.guards.delta_min_ms:
                    break

                if self.cfg.sched.fixed_split is not None:
                    f_ga, f_pso, f_rl = self.cfg.sched.fixed_split
                    targets = {"GA": f_ga * self.cfg.B_tau_ms, "PSO": f_pso * self.cfg.B_tau_ms, "RL": f_rl * self.cfg.B_tau_ms}
                    gaps = {m: targets[m] - csum[m] for m in ["GA", "PSO", "RL"]}
                    for m in list(gaps.keys()):
                        if (m == "GA" and not self.enable_ga) or (m == "PSO" and not self.enable_pso) or (m == "RL" and not self.enable_rl):
                            gaps[m] = -1e18
                    m_star = max(gaps, key=gaps.get)
                else:
                    guard_candidates = [m for m in ["GA", "PSO", "RL"]
                                        if n[m] < self.cfg.sched.n_min and self._module_index(m) > -1e17]
                    if len(guard_candidates) > 0:
                        m_star = next(m for m in ["GA", "PSO", "RL"] if m in guard_candidates)
                    else:
                        idxs = {m: self._module_index(m) for m in ["GA", "PSO", "RL"]}
                        m_star = max(idxs, key=idxs.get)

                allowed = min(self.cfg.guards.delta_max_ms, B_rem)
                deadline = now_ms() + allowed
                eval_seeds = [eval_seed_base + 100000 * self.seed + 1000 * tau + k for k in range(eval_cfg.K_eval)]

                step_t0 = now_ms()
                evaluated: List[EvaluatedCandidate] = []
                barJ = 0.0
                se = 0.0
                produced_eval = 0

                if m_star == "GA" and self.enable_ga and self.ga is not None:
                    evaluated, barJ, se, used = self.ga.step(self.env, eval_cfg, eval_seeds, self.tmodel, deadline,
                                                             self.cfg.descriptor_seed, self.cfg.descriptor_H)
                    env_steps_online += int(used)
                    produced_eval = 1 if len(evaluated) > 0 else 0

                elif m_star == "PSO" and self.enable_pso and self.pso is not None:
                    evaluated, barJ, se, used = self.pso.step(self.env, eval_cfg, eval_seeds, self.tmodel, deadline)
                    env_steps_online += int(used)
                    produced_eval = 1 if len(evaluated) > 0 else 0

                elif m_star == "RL" and self.enable_rl and self.rl is not None:
                    train_seed = train_seed_base + 100000 * self.seed + 1000 * tau + n[m_star]
                    evaluated, barJ, se, used, Lema = self.rl.step(self.env, eval_cfg, train_seed, eval_seeds, self.tmodel, deadline)
                    self.L_rl = float(Lema)
                    env_steps_online += int(used)
                    produced_eval = 1 if len(evaluated) > 0 else 0
                else:
                    break

                measured = now_ms() - step_t0
                charged = max(float(measured), float(self.cfg.guards.delta_min_ms))
                overrun = 1 if (charged > allowed + 1e-6) else 0

                B_rem -= charged
                if B_rem < 0.0:
                    B_rem = 0.0

                n[m_star] += 1
                csum[m_star] += charged
                self.Nm[m_star] += 1
                self.Ntot += 1

                for c in evaluated:
                    C_tau.append((c.theta.copy(), float(c.J), float(c.se), m_star))

                if produced_eval == 1:
                    self._update_module_signal_if_eval(m_star, tau, barJ, se, charged)

                rows.append({
                    "tau": tau, "module": m_star, "step_kind": "update",
                    "produced_eval": produced_eval, "charged_ms": charged, "allowed_ms": allowed,
                    "overrun": overrun, "B_rem_ms": B_rem, "barJ_step": float(barJ),
                    "se_step": float(se), "tilde_delta_GA": float(self.signals["GA"].tilde_delta),
                    "tilde_delta_PSO": float(self.signals["PSO"].tilde_delta),
                    "tilde_delta_RL": float(self.signals["RL"].tilde_delta),
                    "D_pop": float(self.D_pop), "L_RL": float(self.L_rl),
                    "c_GA": float(csum["GA"]), "c_PSO": float(csum["PSO"]), "c_RL": float(csum["RL"]),
                    "env_steps_online_partial": int(env_steps_online),
                })

            if div_reserved > 0.0 and self.enable_ga and self.enable_diversity:
                div_deadline = now_ms() + div_reserved
                self.prev_D_pop = self.D_pop
                self.D_pop = float(self._compute_diversity(div_deadline))
                dD = self._dD_last()
                self.window_dD.append(float(dD))
                if len(self.window_dD) > self.cfg.sched.W:
                    self.window_dD = self.window_dD[-self.cfg.sched.W:]
                csum["GA"] += float(div_reserved)
                rows.append({
                    "tau": tau, "module": "GA", "step_kind": "diversity", "charged_ms": float(div_reserved),
                    "D_pop": float(self.D_pop),
                })

            compute_ms = now_ms() - t_comp0
            per_interval_compute_ms.append(float(compute_ms))

            if len(C_tau) == 0:
                if self.enable_pso and self.pso is not None:
                    theta_star = self.pso.gbest.copy()
                elif self.enable_rl and self.rl is not None:
                    theta_star = self.rl.phi.copy()
                elif self.enable_ga and self.ga is not None:
                    theta_star = self.ga.P[0].copy()
                else:
                    theta_star = init_theta(self.rng, self.policy.theta_dim)
            else:
                theta_star = max(C_tau, key=lambda x: x[1])[0]

            if self.enable_ga and self.enable_rl and self.ga is not None and self.rl is not None:
                if self.rng.random() < float(self.cfg.p_inj):
                    fit = getattr(self.ga, "last_fitness", None)
                    if isinstance(fit, list) and len(fit) == len(self.ga.P):
                        worst = int(np.argmin(np.asarray(fit, dtype=float)))
                    else:
                        worst = int(self.rng.integers(0, len(self.ga.P)))
                    self.ga.P[worst] = project_theta(self.rl.phi.copy())

            if self.enable_rl and self.rl is not None:
                obs_tr, act_tr, _ = rollout_trajectory(self.env, self.policy, theta_star,
                                                      seed=eval_seed_base + 999999 + 100000 * self.seed + 1000 * tau,
                                                      H=eval_cfg.H, deterministic=True)
                self.elite_buf.add(obs_tr, act_tr)

            eval_seeds_deploy = [eval_seed_base + 999999 + 100000 * self.seed + 1000 * tau + k for k in range(eval_cfg.K_eval)]
            returns: List[float] = []
            for s in eval_seeds_deploy:
                G, _ = rollout_return(self.env, self.policy, theta_star, seed=s, H=eval_cfg.H, gamma=eval_cfg.gamma,
                                      deterministic=eval_cfg.deterministic_eval)
                returns.append(G)
            score = float(np.mean(returns)) if returns else 0.0

            per_interval_score.append(float(score))
            per_interval_envsteps_online.append(int(env_steps_online))

        log_df = pd.DataFrame(rows)
        log_path = os.path.join(logs_dir, f"interval_log_seed{self.seed}_{self.method_name}.csv")
        log_df.to_csv(log_path, index=False)

        mu = float(np.mean(per_interval_score)) if len(per_interval_score) else 0.0
        sd = float(np.std(per_interval_score, ddof=1)) if len(per_interval_score) > 1 else 0.0
        T = mu + 0.25 * sd
        
        steps_to_T = 0
        cum_steps = 0
        for sc, st in zip(per_interval_score, per_interval_envsteps_online):
            cum_steps += int(st)
            if sc >= T:
                steps_to_T = cum_steps
                break
        if steps_to_T == 0: steps_to_T = cum_steps

        p95 = percentile(per_interval_compute_ms, 95)
        p99 = percentile(per_interval_compute_ms, 99)

        return {
            "method": self.method_name, "seed": int(self.seed),
            "Score_mean": float(np.mean(per_interval_score)), "Score_std": float(np.std(per_interval_score, ddof=1)),
            "WinPct_proxy": float(np.mean([1.0 if s >= T else 0.0 for s in per_interval_score])) * 100.0,
            "Steps_to_T": int(steps_to_T), "p95_ms": float(p95), "p99_ms": float(p99),
            "intervals": int(intervals), "B_tau_ms": float(self.cfg.B_tau_ms),
            "env_steps_total_online": int(sum(per_interval_envsteps_online)),
        }


class ESRunner:
    def __init__(self, env: Any, policy: LinearSoftmaxPolicy, cfg: DLGPRConfig, seed: int, method_name: str = "ES"):
        self.env = env
        self.policy = policy
        self.cfg = cfg
        self.seed = seed
        self.method_name = method_name
        self.rng = np.random.default_rng(seed)
        self.tmodel = TimeModel()
        self.es = ESBaseline(policy, cfg.es, self.rng)

    def run(self, intervals: int, eval_cfg: EvalConfig,
            eval_seed_base: int, train_seed_base: int,
            out_dir: str) -> Dict[str, float]:
        ensure_dir(out_dir)
        logs_dir = os.path.join(out_dir, "logs")
        figs_dir = os.path.join(out_dir, "figs")
        tables_dir = os.path.join(out_dir, "tables")
        ensure_dir(logs_dir); ensure_dir(figs_dir); ensure_dir(tables_dir)

        per_interval_score: List[float] = []
        per_interval_envsteps_online: List[int] = []
        per_interval_compute_ms: List[float] = []
        rows = []

        for tau in range(1, intervals + 1):
            B_rem = float(self.cfg.B_tau_ms)
            env_steps_online = 0
            C_tau = []
            t_comp0 = now_ms()

            while B_rem > 0.0:
                if B_rem < self.cfg.guards.delta_min_ms:
                    break
                allowed = min(self.cfg.guards.delta_max_ms, B_rem)
                deadline = now_ms() + allowed
                eval_seeds = [eval_seed_base + 100000 * self.seed + 1000 * tau + k for k in range(eval_cfg.K_eval)]
                step_t0 = now_ms()
                evaluated, _, _, used = self.es.step(self.env, eval_cfg, eval_seeds, self.tmodel, deadline)
                env_steps_online += int(used)
                charged = max(float(now_ms() - step_t0), float(self.cfg.guards.delta_min_ms))
                B_rem -= charged
                if B_rem < 0: B_rem = 0.0
                for c in evaluated:
                    C_tau.append((c.theta.copy(), float(c.J), float(c.se)))
                rows.append({"tau": tau, "charged_ms": charged})

            compute_ms = now_ms() - t_comp0
            per_interval_compute_ms.append(float(compute_ms))

            if len(C_tau) == 0: theta_star = self.es.mu.copy()
            else: theta_star = max(C_tau, key=lambda x: x[1])[0]

            eval_seeds_deploy = [eval_seed_base + 999999 + 100000 * self.seed + 1000 * tau + k for k in range(eval_cfg.K_eval)]
            returns = []
            for s in eval_seeds_deploy:
                G, _ = rollout_return(self.env, self.policy, theta_star, seed=s, H=eval_cfg.H, gamma=eval_cfg.gamma, deterministic=eval_cfg.deterministic_eval)
                returns.append(G)
            score = float(np.mean(returns)) if returns else 0.0
            per_interval_score.append(score)
            per_interval_envsteps_online.append(env_steps_online)

        pd.DataFrame(rows).to_csv(os.path.join(logs_dir, f"interval_log_seed{self.seed}_{self.method_name}.csv"), index=False)
        
        # Stats
        mu = float(np.mean(per_interval_score)) if per_interval_score else 0.0
        T = mu + 0.25 * (float(np.std(per_interval_score)) if len(per_interval_score)>1 else 0.0)
        cum = 0
        steps_to_T = 0
        for sc, st in zip(per_interval_score, per_interval_envsteps_online):
            cum += st
            if sc >= T: 
                steps_to_T = cum
                break
        if steps_to_T == 0: steps_to_T = cum

        return {
            "method": self.method_name, "seed": int(self.seed),
            "Score_mean": mu, "Score_std": float(np.std(per_interval_score)),
            "WinPct_proxy": float(np.mean([1.0 if s >= T else 0.0 for s in per_interval_score])) * 100.0,
            "Steps_to_T": int(steps_to_T), "p95_ms": float(percentile(per_interval_compute_ms, 95)),
            "p99_ms": float(percentile(per_interval_compute_ms, 99)),
            "intervals": int(intervals), "B_tau_ms": float(self.cfg.B_tau_ms),
            "env_steps_total_online": int(sum(per_interval_envsteps_online)),
        }


# ============================ Plotting =========================================

def plot_timeline_budget(fig_path: str, B_tau_ms: float = 120.0):
    fig = plt.figure(figsize=(7.2, 1.8))
    ax = plt.gca()
    ax.set_axis_off()
    x0, y0 = 0.05, 0.55
    w, h = 0.90, 0.12
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="none"))
    segs = [0.25, 0.30, 0.25, 0.20]
    labels = ["GA step", "PSO step", "RL step", "Deploy"]
    x = x0
    for s, lab in zip(segs, labels):
        ax.add_patch(FancyBboxPatch((x, y0), w*s, h, boxstyle="round,pad=0.01", linewidth=0.8, facecolor="none"))
        ax.text(x + w*s/2, y0 + h/2, lab, ha="center", va="center", fontsize=9)
        x += w*s
    ax.text(0.5, 0.85, f"Per-interval hard wall-clock budget $B_\\tau$ = {B_tau_ms:.0f} ms", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

def plot_budget_simplex(fig_path: str):
    fig = plt.figure(figsize=(4.4, 3.8))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_axis_off()
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, math.sqrt(3)/2])
    tri = Polygon([A, B, C], closed=True, fill=False, linewidth=1.2)
    ax.add_patch(tri)
    ax.text(A[0]-0.05, A[1]-0.05, "GA", fontsize=10)
    ax.text(B[0]+0.02, B[1]-0.05, "PSO", fontsize=10)
    ax.text(C[0]-0.03, C[1]+0.02, "RL", fontsize=10)
    pts = {"exploration-heavy": (0.70, 0.20, 0.10), "balanced": (0.34, 0.33, 0.33), "exploitation-heavy": (0.20, 0.65, 0.15), "adaptation-heavy": (0.20, 0.15, 0.65)}
    for name, (ga, pso, rl) in pts.items():
        P = ga*A + pso*B + rl*C
        ax.plot(P[0], P[1], marker="o")
        ax.text(P[0]+0.01, P[1]+0.01, name, fontsize=8)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, math.sqrt(3)/2 + 0.15)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

def _box(ax, xy, wh, text, fontsize=8):
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.0, facecolor="none"))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def _arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=10, linewidth=1.0))

def plot_arch_dlgpr(fig_path: str):
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = plt.gca()
    ax.set_axis_off()
    _box(ax, (0.05, 0.60), (0.18, 0.18), "Shared codec\nEnc/Dec", 9)
    _box(ax, (0.27, 0.60), (0.18, 0.18), "Projection/\nRepair $\\Pi_\\Theta$", 9)
    _box(ax, (0.49, 0.60), (0.18, 0.18), "Eval operator\n$\\widehat{J}$", 9)
    _box(ax, (0.72, 0.70), (0.23, 0.12), "Candidate pool\n$\\mathcal{C}_\\tau$", 9)
    _box(ax, (0.72, 0.52), (0.23, 0.12), "Deploy\n$\\theta_\\tau^\\star$", 9)
    _box(ax, (0.10, 0.30), (0.18, 0.14), "GA\n(Exploration)", 9)
    _box(ax, (0.35, 0.30), (0.18, 0.14), "PSO\n(Exploitation)", 9)
    _box(ax, (0.60, 0.30), (0.18, 0.14), "RL\n(Adaptation)", 9)
    _arrow(ax, (0.23, 0.69), (0.27, 0.69))
    _arrow(ax, (0.45, 0.69), (0.49, 0.69))
    _arrow(ax, (0.67, 0.69), (0.72, 0.76))
    _arrow(ax, (0.19, 0.37), (0.27, 0.60))
    _arrow(ax, (0.44, 0.37), (0.36, 0.60))
    _arrow(ax, (0.69, 0.37), (0.58, 0.60))
    _arrow(ax, (0.83, 0.70), (0.83, 0.64))
    ax.text(0.60, 0.16, "Injection / Distillation handshakes", fontsize=9, ha="center")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

def plot_candidate_lifecycle(fig_path: str):
    fig = plt.figure(figsize=(7.2, 3.4))
    ax = plt.gca()
    ax.set_axis_off()
    _box(ax, (0.05, 0.62), (0.18, 0.14), "GA produces\ncandidates", 9)
    _box(ax, (0.05, 0.40), (0.18, 0.14), "PSO produces\ncandidates", 9)
    _box(ax, (0.05, 0.18), (0.18, 0.14), "RL produces\n$\\theta^{RL}_\\tau$", 9)
    _box(ax, (0.32, 0.40), (0.20, 0.18), "Projection/\nRepair", 9)
    _box(ax, (0.58, 0.40), (0.20, 0.18), "Aggregate into\n$\\mathcal{C}_\\tau$", 9)
    _box(ax, (0.83, 0.40), (0.14, 0.18), "Select\n$\\theta^\\star_\\tau$", 9)
    _arrow(ax, (0.23, 0.69), (0.32, 0.49))
    _arrow(ax, (0.23, 0.47), (0.32, 0.49))
    _arrow(ax, (0.23, 0.25), (0.32, 0.49))
    _arrow(ax, (0.52, 0.49), (0.58, 0.49))
    _arrow(ax, (0.78, 0.49), (0.83, 0.49))
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

def plot_results_traces(log_df: pd.DataFrame, out_dir: str, title_prefix: str):
    figs_dir = os.path.join(out_dir, "figs")
    ensure_dir(figs_dir)
    if "c_GA" in log_df.columns:
        last_rows = log_df.sort_values(["tau"]).groupby("tau").tail(1)
        tau = last_rows["tau"].to_numpy()
        B_tau = float(last_rows["B_tau_ms"].iloc[0]) if "B_tau_ms" in last_rows.columns and len(last_rows) else 1.0
        c_ga = last_rows["c_GA"].to_numpy()
        c_pso = last_rows["c_PSO"].to_numpy()
        c_rl = last_rows["c_RL"].to_numpy()
        plt.figure()
        plt.plot(tau, c_ga / (B_tau + 1e-12), label="GA share")
        plt.plot(tau, c_pso / (B_tau + 1e-12), label="PSO share")
        plt.plot(tau, c_rl / (B_tau + 1e-12), label="RL share")
        plt.xlabel("Interval $\\tau$")
        plt.ylabel("Allocation share")
        plt.title(f"{title_prefix} - Allocation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"alloc_share_{title_prefix}.png"), bbox_inches="tight")
        plt.close()

def generate_paper_figures(figs_dir: str, B_tau_ms: float):
    ensure_dir(figs_dir)
    plot_timeline_budget(os.path.join(figs_dir, "timeline_budget.pdf"), B_tau_ms=B_tau_ms)
    plot_budget_simplex(os.path.join(figs_dir, "budget_simplex.pdf"))
    plot_arch_dlgpr(os.path.join(figs_dir, "arch_dlgpr.pdf"))
    plot_candidate_lifecycle(os.path.join(figs_dir, "candidate_lifecycle.pdf"))

# ============================ Tables (CSV + LaTeX) ==============================

def df_to_booktabs_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    colspec = "l" + "c" * (len(cols) - 1)
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float): vals.append(f"{v:.3f}")
            else: vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def aggregate_summaries(summaries: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(summaries)
    agg = df.groupby("method").agg({
        "Score_mean": ["mean", "std"], "WinPct_proxy": ["mean", "std"],
        "Steps_to_T": ["mean", "std"], "p95_ms": ["mean", "std"], "p99_ms": ["mean", "std"],
    })
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    out = pd.DataFrame({
        "Method": agg["method"], "Score": agg["Score_mean_mean"].round(3),
        "Win%": agg["WinPct_proxy_mean"].round(2), "Steps-to-T": agg["Steps_to_T_mean"].round(0).astype(int),
        "p95 ms": agg["p95_ms_mean"].round(2), "p99 ms": agg["p99_ms_mean"].round(2),
    })
    return out

# ============================ Experiment Orchestrator ===========================

def run_method_across_seeds(method_ctor, seeds: List[int], intervals: int,
                            eval_cfg: EvalConfig, env_name: Optional[str],
                            out_dir: str, run_name: str) -> List[Dict[str, float]]:
    summaries: List[Dict[str, float]] = []
    for s in seeds:
        env, obs_dim, nA = make_env(env_name, horizon=eval_cfg.H, seed=s)
        policy = LinearSoftmaxPolicy(obs_dim=obs_dim, n_actions=nA)
        try:
            runner, method_name = method_ctor(env, policy, s)
            method_out_dir = os.path.join(out_dir, run_name, method_name)
            ensure_dir(method_out_dir)
            summary = runner.run(intervals=intervals, eval_cfg=eval_cfg,
                                 eval_seed_base=20240000, train_seed_base=20250000,
                                 out_dir=method_out_dir)
            summaries.append(summary)
            log_path = os.path.join(method_out_dir, "logs", f"interval_log_seed{s}_{method_name}.csv")
            if os.path.exists(log_path):
                plot_results_traces(pd.read_csv(log_path), method_out_dir, title_prefix=f"{method_name}_seed{s}")
        finally:
            if hasattr(env, "close"):
                try: env.close()
                except Exception: pass
    return summaries

def make_methods(cfg_base: DLGPRConfig):
    def ctor_dlgpr(env, policy, seed):
        cfg = dataclasses.replace(cfg_base)
        r = DLGPRRunner(env, policy, cfg, seed, enable_ga=True, enable_pso=True, enable_rl=True, enable_diversity=True, method_name="DLGPR")
        return r, "DLGPR"
    def ctor_ga(env, policy, seed):
        cfg = dataclasses.replace(cfg_base)
        r = DLGPRRunner(env, policy, cfg, seed, enable_ga=True, enable_pso=False, enable_rl=False, enable_diversity=True, method_name="GA-only")
        return r, "GA-only"
    def ctor_pso(env, policy, seed):
        cfg = dataclasses.replace(cfg_base)
        r = DLGPRRunner(env, policy, cfg, seed, enable_ga=False, enable_pso=True, enable_rl=False, enable_diversity=False, method_name="PSO-only")
        return r, "PSO-only"
    def ctor_rl(env, policy, seed):
        cfg = dataclasses.replace(cfg_base)
        r = DLGPRRunner(env, policy, cfg, seed, enable_ga=False, enable_pso=False, enable_rl=True, enable_diversity=False, method_name="RL-only")
        return r, "RL-only"
    def ctor_es(env, policy, seed):
        cfg = dataclasses.replace(cfg_base)
        r = ESRunner(env, policy, cfg, seed, method_name="ES")
        return r, "ES"
    
    return {"DLGPR": ctor_dlgpr, "GA-only": ctor_ga, "PSO-only": ctor_pso, "RL-only": ctor_rl, "ES": ctor_es}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default=None, help="Gym env name.")
    ap.add_argument("--run_name", type=str, default="run_complete")
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--intervals", type=int, default=120)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--K_eval", type=int, default=6)
    ap.add_argument("--B_tau_ms", type=float, default=120.0)
    ap.add_argument("--methods", type=str, default="DLGPR,GA-only,RL-only", help="Methods to run")
    args = ap.parse_args()

    ensure_dir(args.out)
    run_root = os.path.join(args.out, args.run_name)
    ensure_dir(run_root)

    eval_cfg = EvalConfig(H=args.H, K_eval=args.K_eval, gamma=0.99, deterministic_eval=True)
    guards = BudgetGuards(delta_min_ms=5.0, delta_max_ms=30.0)
    cfg_base = DLGPRConfig(B_tau_ms=float(args.B_tau_ms), guards=guards, sched=SchedulerConfig())

    generate_paper_figures(os.path.join(run_root, "paper_figs"), B_tau_ms=float(args.B_tau_ms))

    seeds = list(range(args.seeds))
    method_map = make_methods(cfg_base)
    wanted = [m.strip() for m in args.methods.split(",") if m.strip() in method_map]

    all_summaries = []
    for m in wanted:
        print(f"[RUN] {m} ...")
        s = run_method_across_seeds(method_map[m], seeds, args.intervals, eval_cfg, args.env, args.out, args.run_name)
        all_summaries.extend(s)

    tables_dir = os.path.join(run_root, "tables")
    ensure_dir(tables_dir)
    main_df = aggregate_summaries(all_summaries)
    main_df.to_csv(os.path.join(tables_dir, "mainresults.csv"), index=False)
    
    # Generate LaTeX table
    tex = df_to_booktabs_tex(main_df, caption="Performance results", label="tab:results")
    with open(os.path.join(tables_dir, "mainresults.tex"), "w") as f: f.write(tex)

    print("\n[OK] Run Complete. Check outputs in:", run_root)

if __name__ == "__main__":
    main()