# AdvShape port for IIG-RL-Benchmark

This branch adds Century-RL's asymmetric learner/teacher self-play strategy to
`IIG-RL-Benchmark` as `algorithm=advshape`.

## Design

The benchmark's PPO MLP (`PPOAgent`) is reused unchanged. The self-play
training dynamics are changed to:

- two independent policies: deployed **agent** and auxiliary **environment/teacher**;
- deterministic seat rotation after every completed game;
- PPO updates on each strategy's own decision trajectory;
- batch win-rate advantage scaling:
  - agent factor = `agent_threshold - agent_win_rate`;
  - teacher factor = `environment_threshold - environment_win_rate`;
- separate adaptive perplexity targets for agent and teacher;
- adaptive reverse-KL trust region to the rollout policy;
- only the **agent** is evaluated for exact exploitability and saved as
  `agent.pth`; the teacher is also saved as `environment.pth` for analysis.

The full-method defaults are the current Connect Four recipe from Century-RL
where they map cleanly to this benchmark. They are intentionally exposed as
Hydra knobs rather than hidden in the implementation.

## Smoke run

```bash
python main.py \
  algorithm=advshape \
  game=abrupt_phantom_ttt \
  max_steps=100000 \
  compute_exploitability=True \
  compute_exploitability_every=50000
```

## Benchmark run

```bash
python main.py \
  algorithm=advshape \
  game=abrupt_phantom_ttt \
  max_steps=10000000 \
  compute_exploitability=True \
  compute_exploitability_every=500000 \
  seed=0
```

Exact exploitability is supported for:

- `classical_phantom_ttt`
- `abrupt_phantom_ttt`
- `classical_dark_hex`
- `abrupt_dark_hex`

The resulting `exploitability.csv` is directly comparable to the other
benchmark algorithms because it uses the benchmark's unchanged exact
exploitability callback.

## HPO-matched structural controls

These are the cleanest tests of the learner/teacher hypothesis.

For each exact-exploitability game, `configs/control/{ppo,mmd}/<game>.yaml`
reuses the benchmark authors' **best hyperparameters selected for minimum final
exploitability** wherever AdvShape exposes the same knob. This includes the
optimizer scalar settings, rollout/minibatch sizes, update epochs, gamma,
GAE/value lambda, PPO clipping, value loss settings, gradient clipping,
entropy coefficient, MMD KL coefficient, and the native PPO/MMD learning-rate
annealing schedule.

The matched controls use the benchmark runner's linear schedule exactly:

`lr = initial_lr * max(0, 1 - update / num_updates)`

with `num_updates = max_steps // (num_envs * num_steps) + 1`, matching the PPO
and MMD implementation.

The adaptive PPL and KL controllers are disabled. The entropy coefficient is
the same for learner and teacher. The remaining intended algorithmic changes
are the learner/teacher split and win-rate advantage shaping.

Example, PPO-matched control on Abrupt Phantom TTT:

```bash
python main.py \
  algorithm=advshape \
  +control=ppo/abrupt_phantom_ttt \
  game=abrupt_phantom_ttt \
  max_steps=10000000 \
  compute_exploitability=True
```

MMD-matched control on the same game:

```bash
python main.py \
  algorithm=advshape \
  +control=mmd/abrupt_phantom_ttt \
  game=abrupt_phantom_ttt \
  max_steps=10000000 \
  compute_exploitability=True
```

Use the matching control name for the other three games.

Published selected regularization coefficients copied into these controls:

| game | PPO entropy | MMD entropy | MMD reverse KL |
| --- | ---: | ---: | ---: |
| classical Phantom TTT | 0.05 | 0.05 | 0.05 |
| abrupt Phantom TTT | 0.20 | 0.20 | 0.20 |
| classical Dark Hex | 0.05 | 0.20 | 0.025 |
| abrupt Dark Hex | 0.05 | 0.05 | 0.10 |

## Mechanism ablations

These configs live in `configs/ablation/` and keep the same AdvShape code path.
They are useful after the HPO-matched controls establish the structural result.

### Fixed KL coefficient

```bash
python main.py algorithm=advshape +ablation=advshape_fixed_kl \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### No KL

```bash
python main.py algorithm=advshape +ablation=advshape_no_kl \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### Fixed asymmetric entropy

Disable the perplexity thermostat while retaining the full recipe's fixed
entropy coefficients (`0.1` learner, `0.5` teacher by default):

```bash
python main.py algorithm=advshape +ablation=advshape_fixed_ppl \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### Both controllers fixed

```bash
python main.py algorithm=advshape +ablation=advshape_fixed_controls \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### Core asymmetric recipe

No KL penalty and no adaptive perplexity controller. The learner/teacher split,
win-rate advantage shaping and fixed asymmetric entropy strengths remain:

```bash
python main.py algorithm=advshape +ablation=advshape_core \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

Together, the controls and ablations separate the claims:

1. learner/teacher self-play + difficulty shaping improves on matched PPO/MMD;
2. asymmetric fixed exploration provides an additional gain;
3. adaptive PPL/KL control provides any further gain.

For the paper, the HPO-matched controls should be treated as the primary
structural comparison. The full AdvShape recipe is then a separate tuned
version of the method, avoiding post-hoc claims that the structural gain came
from more favorable regularization coefficients or learning-rate annealing.
