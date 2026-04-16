#!/usr/bin/env bash
set -euo pipefail

# Grid runner for the Pelotas election ABM.
# It varies selected parameters and stores one folder per simulation.
#
# Usage:
#   bash run_pelotas_abm_grid.sh
#
# Edit the arrays below to define the parameter grid.

PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT=${SCRIPT:-./pelotas_abm.py}
BASE_OUT=${BASE_OUT:-/media/camafeu/data/rossatto/election_work/abm_runs}

mkdir -p "$BASE_OUT"

# Fixed parameters
CYCLE=${CYCLE:-2024}
N_VOTERS=${N_VOTERS:-20000}
N_CANDIDATES=${N_CANDIDATES:-300}
N_PARTIES=${N_PARTIES:-12}
N_BAIRROS=${N_BAIRROS:-12}
N_STEPS=${N_STEPS:-12}
N_SEATS=${N_SEATS:-21}

# Optional empirical files. Leave empty to generate synthetic populations.
PARTY_CSV=${PARTY_CSV:-}
CANDIDATE_CSV=${CANDIDATE_CSV:-}

# Parameter grid
SEEDS=(101 102 103)
BETA_VISIBILITY=(0.8 1.0 1.2)
DELTA_SOCIAL=(0.5 0.85 1.2)
ETA_PARTY=(0.6 0.95 1.3)
F_STRATEGIC=(0.0 0.35 0.7)
PARTY_CONCENTRATION=(0.8 1.2 1.8)

# You can add more arrays if needed, but note that the total number of runs is the
# product of all array lengths.

echo "Starting grid..."
run_id=0
for seed in "${SEEDS[@]}"; do
  for beta in "${BETA_VISIBILITY[@]}"; do
    for delta in "${DELTA_SOCIAL[@]}"; do
      for eta in "${ETA_PARTY[@]}"; do
        for fstr in "${F_STRATEGIC[@]}"; do
          for phi in "${PARTY_CONCENTRATION[@]}"; do
            run_id=$((run_id + 1))
            outdir="$BASE_OUT/run_${run_id}_seed${seed}_beta${beta}_delta${delta}_eta${eta}_fstr${fstr}_phi${phi}"
            mkdir -p "$outdir"

            cmd=(
              "$PYTHON_BIN" "$SCRIPT"
              --output-dir "$outdir"
              --seed "$seed"
              --cycle "$CYCLE"
              --n-voters "$N_VOTERS"
              --n-candidates "$N_CANDIDATES"
              --n-parties "$N_PARTIES"
              --n-bairros "$N_BAIRROS"
              --n-seats "$N_SEATS"
              --n-steps "$N_STEPS"
              --beta-visibility "$beta"
              --delta-social "$delta"
              --eta-party "$eta"
              --f-strategic "$fstr"
              --party-concentration-mean "$phi"
            )

            if [[ -n "$PARTY_CSV" ]]; then
              cmd+=(--party-csv "$PARTY_CSV")
            fi
            if [[ -n "$CANDIDATE_CSV" ]]; then
              cmd+=(--candidate-csv "$CANDIDATE_CSV")
            fi

            echo "[$run_id] ${cmd[*]}"
            "${cmd[@]}" > "$outdir/stdout.json"
          done
        done
      done
    done
  done
done

echo "Grid finished. Consolidating summaries..."

echo "run_dir,cycle,seed,n_voters,n_candidates,n_parties,n_seats,electoral_quotient,total_valid_votes,elected_total,gini_votes,top10_vote_share,corr_resources_votes,corr_visibility_votes,largest_party_vote_share,largest_party_seats,n_parties_with_seats" > "$BASE_OUT/grid_summary.csv"

while IFS= read -r -d '' file; do
  run_dir=$(dirname "$file")
  python3 - <<PY >> "$BASE_OUT/grid_summary.csv"
import json, os
file = ${file@Q}
run_dir = ${run_dir@Q}
with open(file, 'r', encoding='utf-8') as fh:
    d = json.load(fh)
vals = [
    run_dir,
    d.get('cycle',''), d.get('seed',''), d.get('n_voters',''), d.get('n_candidates',''),
    d.get('n_parties',''), d.get('n_seats',''), d.get('electoral_quotient',''),
    d.get('total_valid_votes',''), d.get('elected_total',''), d.get('gini_votes',''),
    d.get('top10_vote_share',''), d.get('corr_resources_votes',''),
    d.get('corr_visibility_votes',''), d.get('largest_party_vote_share',''),
    d.get('largest_party_seats',''), d.get('n_parties_with_seats','')
]
print(','.join(str(x) for x in vals))
PY
done < <(find "$BASE_OUT" -name summary.json -print0 | sort -z)

echo "Consolidated file: $BASE_OUT/grid_summary.csv"
