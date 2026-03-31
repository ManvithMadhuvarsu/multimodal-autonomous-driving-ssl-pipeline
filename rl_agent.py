"""
=============================================================================
08_rl_agent.py — PPO Reinforcement Learning Agent + Results Aggregation
=============================================================================
No structural bugs found in original. This version adds:
  - Proper import paths (setup/models instead of numeric prefixes)
  - aggregate_test_results correctly imported from here by run_pipeline
=============================================================================
"""

import math
import os
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from setup import OUTPUT_ROOT, RL_DIR, DEVICE, logger, read_json, write_json
from models import FullADModel

# Paths
AGG_OUT   = OUTPUT_ROOT / "aggregated_metrics.json"
CSV_OUT   = OUTPUT_ROOT / "aggregated_metrics.csv"
AGG_FLAG  = OUTPUT_ROOT / "cell20_complete.flag"
RL_CKPT_DIR = RL_DIR


# ─────────────────────────────────────────────
# Reward Shaping
# ─────────────────────────────────────────────
def compute_reward(state: dict) -> float:
    """
    Shaped reward for autonomous driving.
    state keys: collision (bool), lane_offset (float), speed (float),
                progress (float), comfort (float)
    """
    r = 0.0
    if state.get("collision", False):
        return -10.0
    # Progress reward
    r += state.get("progress", 0.0) * 2.0
    # Lane keeping (penalise offset)
    lane_off = abs(state.get("lane_offset", 0.0))
    r -= min(lane_off * 0.5, 1.0)
    # Speed (reward comfortable speed, penalise extremes)
    speed = state.get("speed", 0.0)
    r += max(0.0, 1.0 - abs(speed - 8.0) / 8.0)
    # Comfort (jerk minimisation)
    r -= state.get("jerk", 0.0) * 0.1
    return float(r)


# ─────────────────────────────────────────────
# Rollout Buffer (GAE)
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards   = [], [], []
        self.log_probs, self.values, self.dones   = [], [], []

    def clear(self):
        self.__init__()

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95,
                        last_value: float = 0.0):
        """Generalised Advantage Estimation (GAE)."""
        n   = len(self.rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        vals = self.values + [last_value]
        for t in reversed(range(n)):
            delta  = self.rewards[t] + gamma * vals[t+1] * (1 - self.dones[t]) - vals[t]
            gae    = delta + gamma * lam * (1 - self.dones[t]) * gae
            adv[t] = gae
        returns = adv + np.array(self.values, dtype=np.float32)
        return torch.tensor(adv,     dtype=torch.float32), \
               torch.tensor(returns, dtype=torch.float32)


# ─────────────────────────────────────────────
# PPO Actor-Critic Network
# ─────────────────────────────────────────────
class PPOPolicy(nn.Module):
    """Actor-Critic operating on 512-D state embedding from FullADModel."""
    def __init__(self, state_dim: int = 512, action_dim: int = 4):
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.GELU(),
        )
        self.actor  = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        h   = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, x: torch.Tensor):
        logits, value = self(x)
        dist    = torch.distributions.Categorical(logits=logits)
        action  = dist.sample()
        log_p   = dist.log_prob(action)
        return action.item(), log_p.item(), value.item()


# ─────────────────────────────────────────────
# Wrapper: FullADModel → RL state vector
# ─────────────────────────────────────────────
class WrapperModel(nn.Module):
    def __init__(self, ad_model: FullADModel):
        super().__init__()
        self.ad = ad_model

    @torch.no_grad()
    def get_state(self, rgb=None, thermal=None, lidar=None, radar=None) -> torch.Tensor:
        out = self.ad(rgb=rgb, thermal=thermal, lidar=lidar, radar=radar)
        return out["gnn"]   # 512-D scene-graph embedding


# ─────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────
class PPOTrainer:
    def __init__(self, policy: PPOPolicy, lr: float = 3e-4,
                 clip_eps: float = 0.2, epochs: int = 4,
                 entropy_coef: float = 0.01, vf_coef: float = 0.5):
        self.policy      = policy
        self.opt         = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
        self.clip_eps    = clip_eps
        self.ppo_epochs  = epochs
        self.entropy_c   = entropy_coef
        self.vf_c        = vf_coef

    def update(self, buffer: RolloutBuffer):
        states    = torch.stack(buffer.states).to(DEVICE)
        actions   = torch.tensor(buffer.actions, dtype=torch.long, device=DEVICE)
        old_lp    = torch.tensor(buffer.log_probs, dtype=torch.float32, device=DEVICE)
        advantages, returns = buffer.compute_returns()
        advantages = advantages.to(DEVICE)
        returns    = returns.to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            logits, values = self.policy(states)
            dist     = torch.distributions.Categorical(logits=logits)
            new_lp   = dist.log_prob(actions)
            entropy  = dist.entropy().mean()
            ratio    = (new_lp - old_lp).exp()

            # Clipped surrogate objective
            surr1    = ratio * advantages
            surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_l  = -torch.min(surr1, surr2).mean()
            critic_l = ((values - returns) ** 2).mean()
            loss     = actor_l + self.vf_c * critic_l - self.entropy_c * entropy

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.opt.step()


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────
def save_rl_checkpoint(policy: PPOPolicy, epoch: int, extra: dict = None):
    p = RL_CKPT_DIR / f"ppo_epoch_{epoch}.pth"
    torch.save({"epoch": epoch, "policy_state": policy.state_dict(),
                "extra": extra or {}}, str(p))
    # Keep last 2
    old = sorted(RL_CKPT_DIR.glob("ppo_epoch_*.pth"),
                 key=lambda f: f.stat().st_mtime)[:-2]
    for f in old:
        try: f.unlink()
        except: pass


def load_rl_checkpoint(policy: PPOPolicy) -> int:
    ckpts = sorted(RL_CKPT_DIR.glob("ppo_epoch_*.pth"),
                   key=lambda f: f.stat().st_mtime)
    if not ckpts:
        return 0
    try:
        data = torch.load(str(ckpts[-1]), map_location=DEVICE, weights_only=False)
        policy.load_state_dict(data["policy_state"])
        ep = data.get("epoch", 0)
        logger.info(f"Resumed PPO from {ckpts[-1].name} (ep {ep})")
        return ep + 1
    except Exception as e:
        logger.warning(f"PPO ckpt load: {e}")
        return 0


# ─────────────────────────────────────────────
# Train PPO (call with your gym-compatible env)
# ─────────────────────────────────────────────
def train_ppo(env, full_model: FullADModel, num_epochs: int = 100,
              steps_per_epoch: int = 2048):
    """
    Train PPO agent.
    env: gym-compatible environment returning (obs_dict, reward, done, info)
         obs_dict should have keys: "rgb", "thermal", "lidar", "radar" (file paths)
    """
    policy  = PPOPolicy(state_dim=512, action_dim=4).to(DEVICE)
    trainer = PPOTrainer(policy)
    wrapper = WrapperModel(full_model).to(DEVICE)
    buffer  = RolloutBuffer()
    start_ep = load_rl_checkpoint(policy)

    logger.info(f"PPO training: {num_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch")

    for ep in range(start_ep, num_epochs):
        obs = env.reset()
        buffer.clear()
        ep_reward = 0.0

        for _ in range(steps_per_epoch):
            # Build state from sensor observations
            from perception_heads_and_export import (
                _load_rgb_tensor, _load_thermal_tensor, _load_points_tensor
            )
            rgb     = _load_rgb_tensor(obs.get("rgb"))
            thermal = _load_thermal_tensor(obs.get("thermal"))
            lidar   = _load_points_tensor(obs.get("lidar"))
            radar   = _load_points_tensor(obs.get("radar"))

            state = wrapper.get_state(rgb, thermal, lidar, radar)
            action, log_p, value = policy.act(state.unsqueeze(0))

            obs_next, reward, done, info = env.step(action)
            shaped = compute_reward(info)
            buffer.add(state.cpu(), action, shaped, log_p, value, float(done))
            ep_reward += shaped
            obs = obs_next
            if done:
                obs = env.reset()

        trainer.update(buffer)
        save_rl_checkpoint(policy, ep)
        logger.info(f"PPO ep {ep+1}/{num_epochs} | reward: {ep_reward:.2f}")

    logger.info("PPO training complete.")
    return policy


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def rl_inference(policy: PPOPolicy, state: torch.Tensor) -> int:
    policy.eval()
    with torch.no_grad():
        logits, _ = policy(state.unsqueeze(0).to(DEVICE))
        return torch.argmax(logits, dim=-1).item()


# ─────────────────────────────────────────────
# Results Aggregation
# ─────────────────────────────────────────────
def aggregate_test_results():
    if AGG_FLAG.exists():
        logger.info("Aggregation already done.")
        return

    results_file = OUTPUT_ROOT / "test_results.json"
    if not results_file.exists():
        logger.warning("test_results.json not found — skipping aggregation.")
        return

    import csv, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = read_json(results_file)
    keys    = list(results.keys())
    n       = len(keys)

    det_norms, seg_norms, traj_norms = [], [], []
    for k in tqdm(keys, desc="Aggregating"):
        r = results.get(k, {})
        for lst, norms in [(r.get("detection"),    det_norms),
                           (r.get("segmentation"), seg_norms),
                           (r.get("trajectory"),   traj_norms)]:
            if lst:
                try: norms.append(float(np.linalg.norm(np.array(lst, dtype=float))))
                except: pass

    def _stats(x):
        if not x: return {"count": 0, "mean": None, "std": None}
        a = np.array(x, dtype=float)
        return {"count": int(a.size), "mean": float(a.mean()),
                "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

    agg = {"total_samples": n,
           "detection":    _stats(det_norms),
           "segmentation": _stats(seg_norms),
           "trajectory":   _stats(traj_norms)}
    write_json(AGG_OUT, agg)

    # CSV
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "count", "mean", "std", "min", "max"])
        for head in ("detection", "segmentation", "trajectory"):
            s = agg[head]
            w.writerow([head, s["count"], s.get("mean"), s.get("std"),
                        s.get("min"), s.get("max")])

    # Histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, norms, title in zip(axes,
                                [det_norms, seg_norms, traj_norms],
                                ["Detection", "Segmentation", "Trajectory"]):
        if norms:
            ax.hist(norms, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"{title} Output Norm")
        ax.set_xlabel("L2 Norm"); ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(str(OUTPUT_ROOT / "test_results_hist.png"), dpi=100)
    plt.close(fig)

    AGG_FLAG.touch()
    logger.info(f"Aggregation complete -> {AGG_OUT}")


if __name__ == "__main__":
    aggregate_test_results()
