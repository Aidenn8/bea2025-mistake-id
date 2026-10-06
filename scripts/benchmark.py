"""Time training and inference for one backbone on this machine.

    python scripts/benchmark.py MODEL FORMAT PRECISION MICRO_BATCH

Trains one short epoch (14 optimizer steps of 16 examples, drawn at random from the
development set so input lengths are representative) and predicts 128 examples.
Appends a JSON line to logs/benchmark.jsonl.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mistake_id.data import load_rows  # noqa: E402
from mistake_id.train import Config, Run  # noqa: E402


def main():
    model, fmt, precision, micro = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    aux = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    rows = load_rows("dev")
    pick = np.random.default_rng(0).permutation(len(rows))
    train_rows = [rows[i] for i in pick[:224]]
    pred_rows = [rows[i] for i in pick[224:352]]
    cfg = Config(name="bench", model=model, input_format=fmt, precision=precision, micro_batch=micro,
                 epochs=1, lr=2e-5, aux_weight=aux)
    t0 = time.time()
    run = Run(cfg, log=lambda s: None)
    load_s = time.time() - t0
    run.train(train_rows)
    t1 = time.time()
    probs = run.predict(pred_rows)
    pred_s = time.time() - t1
    peak = torch.mps.driver_allocated_memory() / 2**30 if run.dev.type == "mps" else float("nan")
    rec = {"model": model, "format": fmt, "precision": precision, "micro_batch": micro, "aux": aux,
           "s_per_step": run.train_seconds / 14, "pred_s_per_100": pred_s / 128 * 100, "load_s": load_s,
           "mps_alloc_gb": peak, "finite": bool(np.isfinite(probs).all()),
           "loss": run.history[-1]["train_loss"]}
    print(json.dumps(rec))
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "benchmark.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
