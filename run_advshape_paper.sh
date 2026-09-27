#!/usr/bin/env bash
set -euo pipefail

# Tiny paper-benchmark launcher for AdvShape.
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
#
# Lower exploitability is better.
# Paper references below are approximate visual read-offs of the best generic-PG
# final mean in ICLR 2026 Figure 15. The paper plots the 10-seed mean but does
# not provide those final means as a numeric table. A +/-0.02 dead zone avoids
# pretending the plot has more precision than it does.

CONTROL="${1:-ppo}"
ABLATION="${2:-}"
SEEDS="${SEEDS:-0}"
STEPS="${STEPS:-10000000}"
GAMES="${GAMES:-classical_phantom_ttt abrupt_phantom_ttt classical_dark_hex abrupt_dark_hex}"
GROUP="${GROUP:-advshape_${CONTROL}${ABLATION:+_${ABLATION}}_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-results}"
REF_MARGIN="${REF_MARGIN:-0.02}"

case "$CONTROL" in
  ppo|mmd) ;;
  *) echo "CONTROL must be 'ppo' or 'mmd'" >&2; exit 2 ;;
esac

case "$ABLATION" in
  ""|equal_entropy) ;;
  *) echo "Second argument must be empty or 'equal_entropy'" >&2; exit 2 ;;
esac

# Approximate best generic-PG 10-seed means from ICLR 2026 Fig. 15.
paper_ref() {
  case "$1" in
    classical_dark_hex) echo "0.05" ;;
    abrupt_dark_hex) echo "0.12" ;;
    classical_phantom_ttt) echo "0.10" ;;
    abrupt_phantom_ttt) echo "0.12" ;;
    *) return 1 ;;
  esac
}

green='\033[0;32m'
yellow='\033[0;33m'
red='\033[0;31m'
blue='\033[0;34m'
reset='\033[0m'

score_status() {
  python - "$1" "$2" "$REF_MARGIN" <<'PY'
import sys
score, ref, margin = map(float, sys.argv[1:])
if score < ref - margin:
    print("BETTER")
elif score > ref + margin:
    print("WORSE")
else:
    print("BALLPARK")
PY
}

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

printf "\n${blue}=== AdvShape paper benchmark ===${reset}\n"
printf "control      : %s\n" "$CONTROL"
printf "ablation     : %s\n" "${ABLATION:-none}"
printf "steps        : %s\n" "$STEPS"
printf "seeds        : %s\n" "$SEEDS"
printf "group        : %s\n\n" "$GROUP"

SUMMARY_DIR="$SAVE_DIR/$GROUP"
mkdir -p "$SUMMARY_DIR"
SUMMARY="$SUMMARY_DIR/summary.tsv"
printf "game\tseed\tscore\tpaper_ref_approx\tstatus\trun_dir\n" > "$SUMMARY"

for game in $GAMES; do
  if ! ref=$(paper_ref "$game"); then
    echo "No paper reference registered for game '$game'" >&2
    exit 2
  fi

  printf "\n${blue}--- %s ---${reset}\n" "$game"
  printf "ICLR 2026 best generic-PG final mean: ~%.3f (visual Fig. 15, ±%.3f)\n" "$ref" "$REF_MARGIN"

  game_scores=()
  for seed in $SEEDS; do
    printf "\nseed %s | target: lower than ~%.3f\n" "$seed" "$ref"

    cmd=(
      python main.py
      algorithm=advshape
      "+control=${CONTROL}/${game}"
      "game=${game}"
      "max_steps=${STEPS}"
      "seed=${seed}"
      "save_dir=${SAVE_DIR}"
      "group_name=${GROUP}"
      compute_exploitability=True
      # Final exploitability is always computed by the runner. Setting the
      # periodic interval above STEPS avoids paying for intermediate traversals.
      "compute_exploitability_every=$((STEPS + 1))"
    )

    if [[ "$ABLATION" == "equal_entropy" ]]; then
      cmd+=("+ablation=advshape_equal_teacher_entropy")
    fi

    printf 'running:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"

    base="$SAVE_DIR/$GROUP/advshape/$game"
    run_dir=$(ls -td "$base"/*/ 2>/dev/null | head -n1 || true)
    run_dir=${run_dir%/}
    [[ -n "$run_dir" ]] || { echo "No run directory found under $base" >&2; exit 1; }
    csv="$run_dir/exploitability.csv"
    [[ -f "$csv" ]] || { echo "Missing $csv" >&2; exit 1; }

    score=$(final_score "$csv")
    status=$(score_status "$score" "$ref")
    game_scores+=("$score")

    case "$status" in
      BETTER) color="$green" ;;
      BALLPARK) color="$yellow" ;;
      WORSE) color="$red" ;;
    esac
    printf "${color}final exploitability = %.6f | %s vs ~%.3f${reset}\n" "$score" "$status" "$ref"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$game" "$seed" "$score" "$ref" "$status" "$run_dir" >> "$SUMMARY"
  done

  mean=$(mean_scores "${game_scores[@]}")
  mean_status=$(score_status "$mean" "$ref")
  case "$mean_status" in
    BETTER) color="$green" ;;
    BALLPARK) color="$yellow" ;;
    WORSE) color="$red" ;;
  esac

  n=${#game_scores[@]}
  printf "${color}mean over %d seed(s) = %.6f | %s vs paper ~%.3f${reset}\n" "$n" "$mean" "$mean_status" "$ref"
  if (( n < 10 )); then
    printf "${yellow}note: paper reference is a 10-seed mean; %d seed(s) is only indicative.${reset}\n" "$n"
  fi
done

printf "\n${blue}Summary written to %s${reset}\n" "$SUMMARY"
printf "Reminder: paper thresholds are approximate plot read-offs, not exact tabulated values.\n"
