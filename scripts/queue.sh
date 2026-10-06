#!/bin/bash
# Run the experiments listed in a queue file, one fold per process, under the thermal guard.
#
#   PY=.venv/bin/python bash scripts/queue.sh [queue.txt]
#
# Each non-comment line is "CONFIG SPLIT [FOLD_SEED]". The file is re-read after every
# fold, so it can be edited while the queue runs. Finished folds are skipped (run_cv.py
# checks for its output), and the queue stops at the first failure.
cd "$(dirname "$0")/.."
PY=${PY:-python}
QUEUE=${1:-queue.txt}
mkdir -p logs

next_job() {
  grep -vE '^\s*(#|$)' "$QUEUE" | while read -r cfg split seed; do
    seed=${seed:-0}
    case "$split" in notebook|full) folds="0" ;; *) folds="0 1 2 3 4" ;; esac
    for k in $folds; do
      if [ ! -f "results/runs/$cfg/$split-s$seed/fold$k.npz" ]; then
        echo "$cfg $split $seed $k"
        return
      fi
    done
  done
}

while true; do
  job=$(next_job)
  if [ -z "$job" ]; then echo "$(date '+%F %T') queue empty"; break; fi
  set -- $job
  echo "$(date '+%F %T') START $1 $2 seed=$3 fold=$4"
  "$PY" tools/thermal/guarded_run.py -- "$PY" scripts/run_cv.py "configs/$1.json" --split "$2" \
      --fold-seed "$3" --folds "$4" >> "logs/$1-$2-s$3.log" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "$(date '+%F %T') FAILED($status) $1 $2 seed=$3 fold=$4 (see logs/$1-$2-s$3.log)"
    break
  fi
  echo "$(date '+%F %T') DONE $1 $2 seed=$3 fold=$4 $(grep -o 'held-out strict F1 [0-9.]*' "logs/$1-$2-s$3.log" | tail -1)"
done
