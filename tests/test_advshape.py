import torch

from algorithms.advshape.advshape import (
    AdaptiveKLController,
    AdvShapePolicy,
    PerplexityController,
    Transition,
    normalize_advantages,
)


def _transition(advantage):
    return Transition(
        obs=torch.zeros(3),
        legal_mask=torch.ones(2, dtype=torch.bool),
        action=torch.tensor(0),
        old_logprob=torch.tensor(0.0),
        old_probs=torch.tensor([0.5, 0.5]),
        value=torch.tensor(0.0),
        advantage=advantage,
    )


def _policy(optimizer_type="adam"):
    return AdvShapePolicy(
        observation_shape=(3,),
        num_actions=2,
        device=torch.device("cpu"),
        learning_rate=1.0,
        lr_schedule="benchmark_linear",
        optimizer_type=optimizer_type,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-5,
        perplexity_start=2.0,
        perplexity_end=1.0,
        entropy_strength=0.1,
        entropy_baseline_ratio=1.0,
        perplexity_adaptation_rate=0.0,
        perplexity_beta=0.98,
        perplexity_deadband=0.02,
        kl_target=0.05,
        kl_strength=0.01,
        kl_adaptation_rate=0.0,
        kl_deadband=0.001,
    )


def test_perplexity_controller_schedule_endpoints():
    controller = PerplexityController(
        start=5.0,
        end=1.0,
        init_strength=0.1,
    )
    assert controller.target(0.0) == 5.0
    assert controller.target(1.0) == 1.0


def test_perplexity_controller_pushes_up_when_policy_is_too_deterministic():
    controller = PerplexityController(
        start=5.0,
        end=1.0,
        init_strength=0.1,
        adaptation_rate=0.1,
    )
    initial = controller.strength
    controller.update(measured_ppl=1.0, progress=0.0)
    assert controller.strength > initial


def test_adaptive_kl_controller_pushes_up_above_target():
    controller = AdaptiveKLController(
        target=0.05,
        init_strength=0.01,
        adaptation_rate=0.01,
    )
    initial = controller.strength
    controller.update(0.2)
    assert controller.strength > initial


def test_benchmark_controls_use_torch_adam():
    policy = _policy("adam")
    assert isinstance(policy.optimizer, torch.optim.Adam)
    assert not isinstance(policy.optimizer, torch.optim.AdamW)


def test_benchmark_linear_lr_matches_ppo_formula_and_ignores_warmup():
    policy = _policy("adam")
    policy.set_lr(progress=0.25, update_idx=0, warmup_updates=20)
    assert policy.optimizer.param_groups[0]["lr"] == 0.75


def test_normalize_advantages_is_zero_mean_unit_sample_std():
    transitions = [_transition(-1.0), _transition(0.0), _transition(1.0)]
    normalize_advantages(transitions)
    normalized = torch.tensor([t.normalized_advantage for t in transitions])
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.std(unbiased=True), torch.tensor(1.0), atol=2e-4)
