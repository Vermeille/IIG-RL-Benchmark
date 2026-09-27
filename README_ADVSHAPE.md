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

## Ablations

Ablation configs live in `configs/ablation/` and are merged directly into the
`algorithm` config. They keep the same code path and only disable the mechanism
being tested.

### Fixed KL coefficient

Keep the reverse-KL regularizer but disable its adaptive controller:

```bash
python main.py algorithm=advshape +ablation=advshape_fixed_kl \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### No KL

Remove the KL regularizer entirely:

```bash
python main.py algorithm=advshape +ablation=advshape_no_kl \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### Fixed asymmetric entropy

Disable the perplexity thermostat while retaining the configured fixed entropy
coefficients (`0.1` for the learner, `0.5` for the teacher by default):

```bash
python main.py algorithm=advshape +ablation=advshape_fixed_ppl \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### Both controllers fixed

Keep both regularizers but disable both adaptive controllers:

```bash
python main.py algorithm=advshape +ablation=advshape_fixed_controls \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

### Core asymmetric method

The strongest structural ablation: no KL penalty and no adaptive perplexity
controller. The learner/teacher split, win-rate advantage shaping and fixed
asymmetric entropy strengths remain:

```bash
python main.py algorithm=advshape +ablation=advshape_core \
  game=abrupt_phantom_ttt max_steps=10000000 compute_exploitability=True
```

This makes the comparison useful for separating three claims:

1. asymmetric learner/teacher self-play + difficulty shaping is sufficient;
2. fixed regularization improves it;
3. adaptive KL/perplexity control provides an additional gain.

## Important experimental note

The default perplexity schedule is **not** expected to be universally optimal
across games because raw perplexity depends on branching factor. For a paper,
keep the tuning budget and search procedure explicit and comparable to the
benchmark baselines rather than silently hand-tuning each game after looking at
test exploitability.
