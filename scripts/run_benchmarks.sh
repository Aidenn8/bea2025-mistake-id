#!/bin/bash
# Time every candidate backbone on this machine (see scripts/benchmark.py).
# Each configuration runs in its own process under the thermal guard.
#
# Findings that shaped this list (16 GB M2, PyTorch MPS):
#   micro-batch 8, no memory cap: 14.7 s/step (11.6 GB allocated, machine swapping)
#   micro-batch 4, 6.4 GB cap:    out of memory
#   micro-batch 2, 6.4 GB cap:    3.6 s/step
#   bf16 autocast:                slower than fp32 on this GPU
# so every backbone is timed at micro-batch 2 in fp32. Large (24-layer) models need
# 6-7 GB for weights, gradients and optimiser state alone and are out of reach.
cd "$(dirname "$0")/.."
PY=${PY:-python}
while read -r model fmt precision micro; do
  [ -z "$model" ] && continue
  echo "=== $model $fmt $precision micro=$micro"
  "$PY" tools/thermal/guarded_run.py -- "$PY" scripts/benchmark.py "$model" "$fmt" "$precision" "$micro" \
    2>&1 | grep -E '^\{|Error|error' || echo "FAILED: $model $fmt $precision"
done <<'EOF'
bert-base-uncased pair fp32 2
roberta-base pair fp32 2
microsoft/deberta-v3-base pair fp32 2
sentence-transformers/all-mpnet-base-v2 pair fp32 2
answerdotai/ModernBERT-base pair fp32 2
EOF
