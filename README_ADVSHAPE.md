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

The defaults are the current Connect Four recipe from Century-RL where they
map cleanly to this benchmark. They are intentionally exposed as Hydra knobs
rather than hidden in the implementation.

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

Repeat for the benchmark's exact-exploitability games and desired seeds:

- `classical_phantom_ttt`
- `abrupt_phantom_ttt`
- `classical_dark_hex`
- `abrupt_dark_hex`

The resulting `exploitability.csv` is directly comparable to the other
benchmark algorithms because it uses the benchmark's unchanged exact
exploitability callback.

## Important experimental note

The default perplexity schedule is **not** expected to be universally optimal
across games because raw perplexity depends on branching factor. For a paper,
keep the tuning budget and search procedure explicit and comparable to the
benchmark baselines rather than silently hand-tuning each game after looking at
test exploitability.
