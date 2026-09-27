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
- a deliberately more exploratory teacher;
- separate adaptive perplexity targets for agent and teacher in the full method;
- adaptive reverse-KL trust region to the rollout policy in the full method;
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

## Exact matched-baseline launcher

`run_advshape_paper.sh` avoids hard-coded visual read-offs from Figure 15.
Instead, for every game and seed it:

1. loads the repository-published `best_hparams.yaml` for PPO or MMD;
2. reruns that baseline with the selected seed and training budget;
3. runs the corresponding AdvShape HPO-matched control with the same seed and budget;
4. reads the final exact `avg_score_response` from each `exploitability.csv`;
5. prints the exact paired delta and percentage exploitability reduction;
6. aggregates means over all requested seeds and writes `summary.tsv`.

This is more expensive than comparing to a number copied from a plot, but the
reference is numeric and reproducible rather than a pixel estimate.

Single-seed comparison:

```bash
bash run_advshape_paper.sh ppo
bash run_advshape_paper.sh mmd
```

Equal-teacher-entropy ablation:

```bash
bash run_advshape_paper.sh ppo equal_entropy
```

Ten seeds:

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9" bash run_advshape_paper.sh ppo
```

The paper's Figure 15 also reports mean exploitability and standard deviation
over 10 seeds for the best-performing hyperparameter set. The exact numeric
final means are not tabulated in the paper, so the launcher deliberately does
not claim to reproduce a Figure 15 endpoint from visual inspection. It instead
performs an exact matched rerun of the repository-published best configuration.

The script writes:

```text
results/<group>/summary.tsv
```

with per-seed baseline score, AdvShape score, delta, relative exploitability
reduction, verdict, and both run directories.

## HPO-matched structural controls

For each exact-exploitability game, `configs/control/{ppo,mmd}/<game>.yaml`
reuses the benchmark authors' published best hyperparameters wherever AdvShape
exposes the same knob. This includes optimizer settings, rollout/minibatch
sizes, update epochs, gamma, GAE/value lambda, PPO clipping, value loss
settings, gradient clipping, learner entropy coefficient, MMD KL coefficient,
and the native PPO/MMD learning-rate annealing schedule.

The matched controls use the benchmark runner's linear LR schedule exactly:

```text
lr = initial_lr * max(0, 1 - update / num_updates)
num_updates = max_steps // (num_envs * num_steps) + 1
```

Adaptive PPL and adaptive KL control are disabled. The learner keeps the
baseline-selected entropy coefficient, while the teacher uses a fixed **5x
higher entropy coefficient**, preserving the structural exploration asymmetry
of the Century-RL recipe (`0.1` learner vs `0.5` teacher) without adding an
adaptive thermostat.

Thus the intentional structural differences from the matched PPO/MMD setup are:

1. separate learner and teacher policies;
2. win-rate controlled advantage shaping;
3. fixed higher teacher exploration.

Exact exploitability is supported for:

- `classical_phantom_ttt`
- `abrupt_phantom_ttt`
- `classical_dark_hex`
- `abrupt_dark_hex`

### Selected entropy / KL coefficients

| game | PPO learner entropy | PPO teacher entropy | MMD learner entropy | MMD teacher entropy | MMD reverse KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical Phantom TTT | 0.05 | 0.25 | 0.05 | 0.25 | 0.05 |
| abrupt Phantom TTT | 0.20 | 1.00 | 0.20 | 1.00 | 0.20 |
| classical Dark Hex | 0.05 | 0.25 | 0.20 | 1.00 | 0.025 |
| abrupt Dark Hex | 0.05 | 0.25 | 0.05 | 0.25 | 0.10 |

## Exploration-asymmetry ablation

To isolate the contribution of higher teacher exploration, compose a matched
control with:

```bash
+ablation=advshape_equal_teacher_entropy
```

For example:

```bash
python main.py \
  algorithm=advshape \
  +control=ppo/abrupt_phantom_ttt \
  +ablation=advshape_equal_teacher_entropy \
  game=abrupt_phantom_ttt \
  max_steps=10000000 \
  compute_exploitability=True
```

## Other mechanism ablations

Composable configs under `configs/ablation/`:

- `advshape_equal_teacher_entropy`: remove fixed teacher exploration asymmetry;
- `advshape_fixed_kl`: fixed KL coefficient, no KL controller;
- `advshape_no_kl`: remove KL entirely;
- `advshape_fixed_ppl`: fixed asymmetric entropy, no PPL thermostat;
- `advshape_fixed_controls`: keep regularizers but freeze both controllers;
- `advshape_core`: no KL and no adaptive PPL; retains learner/teacher split,
  win-rate difficulty shaping, and fixed asymmetric entropy.

For the paper, the HPO-matched controls are the primary structural comparison.
The equal-teacher-entropy ablation isolates exploration asymmetry, while the
full AdvShape recipe is a separate version testing whether adaptive PPL/KL adds
anything beyond the core learner/teacher mechanism.
