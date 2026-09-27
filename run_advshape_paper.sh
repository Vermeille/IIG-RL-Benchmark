#!/usr/bin/env bash
set -euo pipefail

# Paper-style AdvShape launcher with an exact matched baseline rerun.
#
# Usage:
#   bash run_advshape_paper.sh ppo
#   bash run_advshape_paper.sh mmd
#   bash run_advshape_paper.sh ppo equal_entropy
#
# Optional environment variables:
#   SEEDS="0 1 2 3 4 5 6 7 8 9"   # default: 0
#   STEPS=10000000                 # default: 10M
#   GAMES="classical_phantom_ttt abrupt_phantom_ttt ..."
#   GROUP=my_run_name
#   SAVE_DIR=results
#
# Lower exploitability is better.
#
# We intentionally do NOT hard-code visual read-offs from Figure 15. The paper
# plots the 10-seed means but does not tabulate their exact final values. Instead
# this script reruns the corresponding PPO/MMD baseline using the repository's
# published best_hparams.yaml, on the exact same seed and training budget, then
# runs AdvShape and compares the exact final exploitabilities.

CONTROL="${1:-ppo}"
ABLATION="${2:-}"
SEEDS="${SEEDS:-0}"
STEPS="${STEPS:-10000000}"
GAMES="${GAMES:-classical_phantom_ttt abrupt_phantom_ttt classical_dark_hex abrupt_dark_hex}"
GROUP="${GROUP:-advshape_${CONTROL}${ABLATION:+_${ABLATION}}_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-results}"

case "$CONTROL" in
  ppo|mmd) ;;
  *) echo "CONTROL must be 'ppo' or 'mmd'" >&2; exit 2 ;;
esac

case "$ABLATION" in
  ""|equal_entropy) ;;
  *) echo "Second argument must be empty or 'equal_entropy'" >&2; exit 2 ;;
esac

green='\033[0;32m'
yellow='\033[0;33m'
red='\033[0;31m'
blue='\033[0;34m'
reset='\033[0m'

final_score() {
  python - "$1" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("empty exploitability.csv")
print(rows[-1]["avg_score_response"])
PY
}

mean_scores() {
  python - "$@" <<'PY'
import statistics, sys
xs = [float(x) for x in sys.argv[1:]]
print(f"{statistics.mean(xs):.6f}")
PY
}

compare_scores() {
  python - "$1" "$2" <<'PY'
import sys
ours, baseline = map(float, sys.argv[1:])
delta = ours - baseline
if abs(delta) < 1e-12:
    status = "TIED"
elif delta < 0:
    status = "BETTER"
else:
    status = "WORSE"
reduction = 0.0 if baseline == 0 else 100.0 * (baseline - ours) / baseline
print(f"{status}\t{delta:.6f}\t{reduction:.2f}")
PY
}

baseline_overrides() {
  local path="$1"
  python - "$path" <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f)
for key, value in cfg.items():
    if key == "algorithm_name":
        continue
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = "null"
    print(f"algorithm.{key}={value}")
PY
}

latest_run_dir() {
  local base="$1"
  local run_dir
  run_dir=$(ls -td "$base"/*/ 2>/dev/null | head -n1 || true)
  run_dir=${run_dir%/}
  [[ -n "$run_dir" ]] || return 1
  printf '%s\n' "$run_dir"
}

printf "\n${blue}=== AdvShape exact matched-baseline benchmark ===${reset}\n"
printf "baseline algo : %s\n" "$CONTROL"
printf "ablation      : %s\n" "${ABLATION:-none}"
printf "steps         : %s\n" "$STEPS"
printf "seeds         : %s\n" "$SEEDS"
printf "group         : %s\n\n" "$GROUP"
printf "Reference = exact rerun of repository best_hparams on the same seed/budget.\n"
printf "No Figure 15 numbers are guessed. Civilization survives another day.\n"

SUMMARY_DIR="$SAVE_DIR/$GROUP"
BASELINE_GROUP="${GROUP}_baseline"
ADV_GROUP="${GROUP}_advshape"
mkdir -p "$SUMMARY_DIR"
SUMMARY="$SUMMARY_DIR/summary.tsv"
printf "game\tseed\tbaseline_algo\tbaseline_score\tadvshape_score\tdelta\treduction_pct\tstatus\tbaseline_run_dir\tadvshape_run_dir\n" > "$SUMMARY"

for game in $GAMES; do
  hparams="best_hyperparameters/min_final_expl_hparams/${CONTROL}/${game}/best_hparams.yaml"
  [[ -f "$hparams" ]] || { echo "Missing baseline hparams: $hparams" >&2; exit 1; }

  printf "\n${blue}--- %s ---${reset}\n" "$game"
  printf "reference: %s with %s\n" "$CONTROL" "$hparams"

  baseline_scores=()
  adv_scores=()

  for seed in $SEEDS; do
    printf "\n${blue}seed %s: baseline %s${reset}\n" "$seed" "$CONTROL"

    mapfile -t overrides < <(baseline_overrides "$hparams")
    baseline_cmd=(
      python main.py
      "algorithm=${CONTROL}"
      "game=${game}"
      "max_steps=${STEPS}"
      "seed=${seed}"
      "save_dir=${SAVE_DIR}"
      "group_name=${BASELINE_GROUP}"
      compute_exploitability=True
      "compute_exploitability_every=$((STEPS + 1))"
    )
    baseline_cmd+=("${overrides[@]}")

    printf 'running:'
    printf ' %q' "${baseline_cmd[@]}"
    printf '\n'
    "${baseline_cmd[@]}"

    baseline_base="$SAVE_DIR/$BASELINE_GROUP/$CONTROL/$game"
    baseline_run=$(latest_run_dir "$baseline_base") || {
      echo "No baseline run directory found under $baseline_base" >&2
      exit 1
    }
    baseline_csv="$baseline_run/exploitability.csv"
    [[ -f "$baseline_csv" ]] || { echo "Missing $baseline_csv" >&2; exit 1; }
    baseline_score=$(final_score "$baseline_csv")
    baseline_scores+=("$baseline_score")
    printf "reference final exploitability = %.6f\n" "$baseline_score"

    printf "\n${blue}seed %s: AdvShape${reset}\n" "$seed"
    adv_cmd=(
      python main.py
      algorithm=advshape
      "+control=${CONTROL}/${game}"
      "game=${game}"
      "max_steps=${STEPS}"
      "seed=${seed}"
      "save_dir=${SAVE_DIR}"
      "group_name=${ADV_GROUP}"
      compute_exploitability=True
      "compute_exploitability_every=$((STEPS + 1))"
    )
    if [[ "$ABLATION" == "equal_entropy" ]]; then
      adv_cmd+=("+ablation=advshape_equal_teacher_entropy")
    fi

    printf 'running:'
    printf ' %q' "${adv_cmd[@]}"
    printf '\n'
    "${adv_cmd[@]}"

    adv_base="$SAVE_DIR/$ADV_GROUP/advshape/$game"
    adv_run=$(latest_run_dir "$adv_base") || {
      echo "No AdvShape run directory found under $adv_base" >&2
      exit 1
    }
    adv_csv="$adv_run/exploitability.csv"
    [[ -f "$adv_csv" ]] || { echo "Missing $adv_csv" >&2; exit 1; }
    adv_score=$(final_score "$adv_csv")
    adv_scores+=("$adv_score")

    IFS=$'\t' read -r status delta reduction <<< "$(compare_scores "$adv_score" "$baseline_score")"
    case "$status" in
      BETTER) color="$green" ;;
      TIED) color="$yellow" ;;
      WORSE) color="$red" ;;
    esac

    printf "${color}AdvShape %.6f vs %s %.6f | %s | delta=%+.6f | reduction=%+.2f%%%s\n" \
      "$adv_score" "$CONTROL" "$baseline_score" "$status" "$delta" "$reduction" "$reset"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$game" "$seed" "$CONTROL" "$baseline_score" "$adv_score" "$delta" "$reduction" "$status" "$baseline_run" "$adv_run" >> "$SUMMARY"
  done

  baseline_mean=$(mean_scores "${baseline_scores[@]}")
  adv_mean=$(mean_scores "${adv_scores[@]}")
  IFS=$'\t' read -r mean_status mean_delta mean_reduction <<< "$(compare_scores "$adv_mean" "$baseline_mean")"
  case "$mean_status" in
    BETTER) color="$green" ;;
    TIED) color="$yellow" ;;
    WORSE) color="$red" ;;
  esac

  n=${#adv_scores[@]}
  printf "\n${color}MEAN over %d seed(s): AdvShape %.6f vs %s %.6f | %s | delta=%+.6f | reduction=%+.2f%%%s\n" \
    "$n" "$adv_mean" "$CONTROL" "$baseline_mean" "$mean_status" "$mean_delta" "$mean_reduction" "$reset"
  if (( n < 10 )); then
    printf "${yellow}note: useful smoke result only; Figure 15 uses 10 fresh seeds.${reset}\n"
  fi
done

printf "\n${blue}Summary written to %s${reset}\n" "$SUMMARY"
printf "For the paper-style comparison use: SEEDS=\"0 1 2 3 4 5 6 7 8 9\" bash run_advshape_paper.sh %s%s\n" \
  "$CONTROL" "${ABLATION:+ $ABLATION}"
