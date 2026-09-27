import torch

from algorithms.advshape.advshape import (
    AdaptiveKLController,
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


def test_normalize_advantages_is_zero_mean_unit_sample_std():
    transitions = [_transition(-1.0), _transition(0.0), _transition(1.0)]
    normalize_advantages(transitions)
    normalized = torch.tensor([t.normalized_advantage for t in transitions])
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.std(unbiased=True), torch.tensor(1.0), atol=2e-4)
