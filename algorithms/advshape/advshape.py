"""Asymmetric two-policy PPO with win-rate controlled advantage shaping.

This is a benchmark-native port of Century-RL's adversarial-advshape trainer.
The learning agent and its environment/teacher are separate policies. Each is
trained with PPO, but its policy gradient is scaled by

    target_win_rate - observed_batch_win_rate.

The two policies can also target different policy perplexities, allowing the
teacher to stay deliberately more diverse than the deployed learner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import nn, optim

from algorithms.ppo.ppo import PPOAgent


@dataclass
class Transition:
    obs: torch.Tensor
    legal_mask: torch.Tensor
    action: torch.Tensor
    old_logprob: torch.Tensor
    old_probs: torch.Tensor
    value: torch.Tensor
    reward: float = 0.0
    advantage: float = 0.0
    normalized_advantage: float = 0.0
    value_target: float = 0.0


class PerplexityController:
    """Adaptive entropy coefficient targeting exp(policy entropy)."""

    def __init__(
        self,
        *,
        start: float,
        end: float,
        init_strength: float,
        baseline_ratio: float = 0.05,
        adaptation_rate: float = 0.004,
        beta: float = 0.98,
        deadband: float = 0.02,
    ):
        if start < 0 or end < 0:
            raise ValueError("perplexity targets must be non-negative")
        if init_strength <= 0:
            raise ValueError("entropy strength must be positive")
        if not 0 <= baseline_ratio <= 1:
            raise ValueError("baseline_ratio must be in [0, 1]")

        self.start = float(start)
        self.end = float(end)
        self.init_strength = float(init_strength)
        self.baseline_strength = self.init_strength * float(baseline_ratio)
        self.min_strength = self.baseline_strength * 0.1
        self.max_strength = self.init_strength * 10.0
        self.adaptation_rate = float(adaptation_rate)
        self.beta = float(beta)
        self.deadband = float(deadband)
        self.strength = self.init_strength
        self.ema = None

    @staticmethod
    def _cosine_schedule(progress: float) -> float:
        # Match Century-RL's default 1%-99% cosine schedule window.
        if progress <= 0.01:
            return 0.0
        if progress >= 0.99:
            return 1.0
        x = (progress - 0.01) / 0.98
        return 0.5 - 0.5 * math.cos(math.pi * x)

    def target(self, progress: float) -> float:
        s = self._cosine_schedule(float(np.clip(progress, 0.0, 1.0)))
        return self.start * (1.0 - s) + self.end * s

    def update(self, measured_ppl: float, progress: float) -> float:
        if self.ema is None:
            self.ema = float(measured_ppl)
        else:
            self.ema = self.beta * self.ema + (1.0 - self.beta) * float(measured_ppl)

        target = self.target(progress)
        error = target - self.ema
        if error > self.deadband:
            self.strength += self.adaptation_rate * self.init_strength * error
        else:
            relax_rate = 0.5 * self.adaptation_rate
            self.strength += relax_rate * (self.baseline_strength - self.strength)

        self.strength = float(
            np.clip(self.strength, self.min_strength, self.max_strength)
        )
        return self.strength


class AdaptiveKLController:
    """Adaptive reverse-KL penalty matching Century-RL's controller."""

    def __init__(
        self,
        *,
        target: float,
        init_strength: float,
        adaptation_rate: float,
        deadband: float = 0.001,
    ):
        self.target = float(target)
        self.init_strength = float(init_strength)
        self.max_strength = self.init_strength * 100.0
        self.adaptation_rate = float(adaptation_rate)
        self.deadband = float(deadband)
        self.strength = self.init_strength

    def update(self, measured_kl: float) -> float:
        error = float(measured_kl) - self.target
        if error > self.deadband:
            self.strength += self.adaptation_rate * error
        else:
            relax_rate = 0.1 * self.adaptation_rate
            self.strength += relax_rate * (self.init_strength - self.strength)
        self.strength = float(
            np.clip(self.strength, self.init_strength, self.max_strength)
        )
        return self.strength


class AdvShapePolicy:
    """One side of the learner/teacher pair."""

    def __init__(
        self,
        *,
        observation_shape,
        num_actions: int,
        device: torch.device,
        learning_rate: float,
        lr_schedule: str,
        optimizer_type: str,
        weight_decay: float,
        adam_beta1: float,
        adam_beta2: float,
        adam_eps: float,
        perplexity_start: float,
        perplexity_end: float,
        entropy_strength: float,
        entropy_baseline_ratio: float,
        perplexity_adaptation_rate: float,
        perplexity_beta: float,
        perplexity_deadband: float,
        kl_target: float,
        kl_strength: float,
        kl_adaptation_rate: float,
        kl_deadband: float,
    ):
        self.device = device
        self.num_actions = int(num_actions)
        self.lr_schedule = str(lr_schedule)
        self.network = PPOAgent(self.num_actions, observation_shape, device).to(device)
        optimizer_kwargs = dict(
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        if optimizer_type == "adam":
            # Benchmark PPO/MMD use torch.optim.Adam with the default betas and
            # eps=1e-5. HPO-matched controls select this path explicitly.
            self.optimizer = optim.Adam(self.network.parameters(), **optimizer_kwargs)
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(self.network.parameters(), **optimizer_kwargs)
        else:
            raise ValueError(f"unknown optimizer_type: {optimizer_type}")
        self.base_learning_rate = float(learning_rate)
        self.perplexity = PerplexityController(
            start=perplexity_start,
            end=perplexity_end,
            init_strength=entropy_strength,
            baseline_ratio=entropy_baseline_ratio,
            adaptation_rate=perplexity_adaptation_rate,
            beta=perplexity_beta,
            deadband=perplexity_deadband,
        )
        self.kl = AdaptiveKLController(
            target=kl_target,
            init_strength=kl_strength,
            adaptation_rate=kl_adaptation_rate,
            deadband=kl_deadband,
        )

    def set_lr(self, progress: float, update_idx: int, warmup_updates: int = 20):
        if self.lr_schedule == "benchmark_linear":
            # Match PPO/MMD's native anneal_learning_rate(): lr = lr0 * (1-u/U).
            scale = max(0.0, 1.0 - progress)
        elif self.lr_schedule == "advshape":
            if warmup_updates > 0 and update_idx < warmup_updates:
                scale = (update_idx + 1) / warmup_updates
            elif progress <= 0.5:
                scale = 1.0
            else:
                x = min(max((progress - 0.5) / 0.5, 0.0), 1.0)
                scale = 0.5 + 0.5 * math.cos(math.pi * x)
        else:
            raise ValueError(f"unknown lr_schedule: {self.lr_schedule}")
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_learning_rate * scale

    @torch.no_grad()
    def act(self, obs: torch.Tensor, legal_mask: torch.Tensor):
        return self.network.get_action_and_value(obs, legal_actions_mask=legal_mask)

    @torch.no_grad()
    def value(self, obs: torch.Tensor):
        return self.network.get_value(obs).view(-1)

    def save_actor(self, path):
        torch.save(self.network.actor.state_dict(), path)

    def load_actor(self, path):
        self.network.actor.load_state_dict(torch.load(path, map_location=self.device))


class AdvShapeLearner:
    """Owns the two independently optimized PPO policies."""

    def __init__(self, *, observation_shape, num_actions, device, config):
        common = dict(
            observation_shape=observation_shape,
            num_actions=num_actions,
            device=device,
            learning_rate=config.learning_rate,
            lr_schedule=config.lr_schedule,
            optimizer_type=config.optimizer_type,
            weight_decay=config.weight_decay,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            adam_eps=config.adam_eps,
            entropy_baseline_ratio=config.entropy_baseline_ratio,
            perplexity_adaptation_rate=config.perplexity_adaptation_rate,
            perplexity_beta=config.perplexity_beta,
            perplexity_deadband=config.perplexity_deadband,
            kl_target=config.kl_target,
            kl_strength=config.kl_strength,
            kl_adaptation_rate=config.kl_adaptation_rate,
            kl_deadband=config.kl_deadband,
        )
        self.agent = AdvShapePolicy(
            **common,
            perplexity_start=config.agent_ppl_start,
            perplexity_end=config.agent_ppl_end,
            entropy_strength=config.agent_entropy_strength,
        )
        self.environment = AdvShapePolicy(
            **common,
            perplexity_start=config.environment_ppl_start,
            perplexity_end=config.environment_ppl_end,
            entropy_strength=config.environment_entropy_strength,
        )
        self.policies = [self.agent, self.environment]
        self.config = config
        self.device = device

    def _tensorize(self, transitions: list[Transition]):
        obs = torch.stack([t.obs for t in transitions]).to(self.device)
        legal = torch.stack([t.legal_mask for t in transitions]).to(self.device)
        actions = torch.stack([t.action for t in transitions]).long().to(self.device)
        old_logprobs = torch.stack([t.old_logprob for t in transitions]).to(self.device)
        old_probs = torch.stack([t.old_probs for t in transitions]).to(self.device)
        raw_advantages = torch.tensor(
            [t.advantage for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        normalized_advantages = torch.tensor(
            [t.normalized_advantage for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        value_targets = torch.tensor(
            [t.value_target for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.stack([t.value for t in transitions]).to(self.device).view(-1)
        return (
            obs,
            legal,
            actions,
            old_logprobs,
            old_probs,
            raw_advantages,
            normalized_advantages,
            value_targets,
            old_values,
        )

    def optimize(
        self,
        strategy_id: int,
        transitions: list[Transition],
        *,
        advantage_scale: float,
        progress: float,
        lr_progress: float | None = None,
        update_idx: int,
    ) -> dict[str, float]:
        if not transitions:
            return {}

        policy = self.policies[strategy_id]
        policy.set_lr(
            progress if lr_progress is None else lr_progress,
            update_idx,
            self.config.warmup_updates,
        )
        (
            obs,
            legal,
            actions,
            old_logprobs,
            old_probs,
            raw_advantages,
            normalized_advantages,
            value_targets,
            old_values,
        ) = self._tensorize(transitions)

        if self.config.advantage_normalization == "global":
            advantages = normalized_advantages
        elif self.config.advantage_normalization in ("ppo_minibatch", "none"):
            advantages = raw_advantages
        else:
            raise ValueError(
                f"unknown advantage_normalization: {self.config.advantage_normalization}"
            )

        n = len(transitions)
        if self.config.minibatch_mode == "ppo_total_batch":
            # PPO defines minibatch size from the complete rollout batch. We
            # keep that size after splitting samples between learner/teacher,
            # so the two policies together perform roughly PPO's total number
            # of optimizer steps instead of doubling it with half-sized batches.
            total_batch_size = int(self.config.num_envs * self.config.num_steps)
            minibatch_size = max(1, total_batch_size // self.config.num_minibatches)
        elif self.config.minibatch_mode == "per_policy":
            num_minibatches = min(self.config.num_minibatches, n)
            minibatch_size = max(1, math.ceil(n / num_minibatches))
        else:
            raise ValueError(f"unknown minibatch_mode: {self.config.minibatch_mode}")
        indices = np.arange(n)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "perplexity": 0.0,
            "entropy_strength": 0.0,
            "kl": 0.0,
            "kl_strength": 0.0,
            "clip_fraction": 0.0,
            "advantage_scale": float(advantage_scale),
        }
        metric_count = 0

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, minibatch_size):
                mb = torch.as_tensor(
                    indices[start : start + minibatch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                _, new_logprob, entropy, new_value, probs = policy.network.get_action_and_value(
                    obs[mb], legal_actions_mask=legal[mb], action=actions[mb]
                )
                new_value = new_value.view(-1)
                logratio = new_logprob - old_logprobs[mb]
                ratio = logratio.exp()

                mb_adv = advantages[mb]
                if self.config.advantage_normalization == "ppo_minibatch":
                    # Match PPO exactly, then apply AdvShape afterwards so the
                    # WR scale is not normalized away.
                    if mb_adv.numel() > 1:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    else:
                        mb_adv = mb_adv - mb_adv.mean()
                mb_adv = mb_adv * float(advantage_scale)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )
                policy_loss = torch.maximum(pg1, pg2).mean()

                if self.config.clip_vloss:
                    v_unclipped = (new_value - value_targets[mb]).pow(2)
                    v_clipped_pred = old_values[mb] + torch.clamp(
                        new_value - old_values[mb],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_clipped = (v_clipped_pred - value_targets[mb]).pow(2)
                    value_loss = 0.5 * torch.maximum(v_unclipped, v_clipped).mean()
                else:
                    value_loss = 0.5 * (new_value - value_targets[mb]).pow(2).mean()

                action_counts = legal[mb].sum(dim=1)
                nontrivial = action_counts > 1
                if nontrivial.any():
                    measured_ppl = entropy[nontrivial].exp().mean().detach().item()
                else:
                    measured_ppl = 0.0
                entropy_strength = policy.perplexity.update(measured_ppl, progress)

                if self.config.kl_mode == "exact":
                    # Full-distribution reverse KL KL(pi_new || pi_rollout),
                    # used by the native AdvShape recipe.
                    safe_new = probs.clamp_min(1e-12)
                    safe_old = old_probs[mb].clamp_min(1e-12)
                    reverse_kl = (
                        probs * (safe_new.log() - safe_old.log())
                    ).sum(dim=1).mean()
                elif self.config.kl_mode == "mmd_approx":
                    # Match IIG-RL-Benchmark MMD exactly: the sample-based
                    # backward-KL approximation from the PPO importance ratio.
                    reverse_kl = (ratio * logratio - (ratio - 1.0)).mean()
                elif self.config.kl_mode == "none":
                    reverse_kl = torch.zeros((), device=self.device)
                else:
                    raise ValueError(f"unknown kl_mode: {self.config.kl_mode}")
                kl_strength = policy.kl.update(reverse_kl.detach().item())

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - entropy_strength * entropy.mean()
                    + kl_strength * reverse_kl
                )

                policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.network.parameters(), self.config.max_grad_norm
                )
                policy.optimizer.step()

                clipped = (
                    (ratio - 1.0).abs() > self.config.clip_coef
                ).float().mean()
                metrics["policy_loss"] += policy_loss.detach().item()
                metrics["value_loss"] += value_loss.detach().item()
                metrics["entropy"] += entropy.mean().detach().item()
                metrics["perplexity"] += measured_ppl
                metrics["entropy_strength"] += entropy_strength
                metrics["kl"] += reverse_kl.detach().item()
                metrics["kl_strength"] += kl_strength
                metrics["clip_fraction"] += clipped.detach().item()
                metric_count += 1

        if metric_count:
            for key in metrics:
                if key != "advantage_scale":
                    metrics[key] /= metric_count
        return metrics


def normalize_advantages(transitions: Iterable[Transition]) -> None:
    transitions = list(transitions)
    if not transitions:
        return
    values = torch.tensor([t.advantage for t in transitions], dtype=torch.float32)
    mean = values.mean().item()
    # Century-RL uses torch.std() over the whole strategy batch.
    std = values.std(unbiased=True).item() if len(values) > 1 else 0.0
    for transition in transitions:
        transition.normalized_advantage = (transition.advantage - mean) / (std + 1e-4)
