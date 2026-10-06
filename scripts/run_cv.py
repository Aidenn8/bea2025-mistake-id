"""Train one configuration under a split scheme and save out-of-fold and test predictions.

    python scripts/run_cv.py configs/NAME.json --split grouped [--fold-seed 0] [--folds 0 1 ...]

For every held-out fold the model is trained on the remaining development data, then
predicts (a) the held-out fold and (b) the full hidden test set. Results go to
results/runs/NAME/SPLIT-sSEED/foldK.npz (+ .json metadata). Finished folds are skipped,
so an interrupted run can simply be restarted.

``--split notebook`` replays the original submission's single 80/20 split (one fold).
``--split full`` trains on the whole development set and predicts only the test set.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mistake_id.data import load_rows  # noqa: E402
from mistake_id.folds import fold_ids, make_folds  # noqa: E402
from mistake_id.metrics import scores  # noqa: E402
from mistake_id.train import Config, Run  # noqa: E402


def versions():
    import sklearn
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "sklearn": sklearn.__version__,
            "machine": platform.machine(), "system": platform.platform()}


def save_atomic(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="grouped", choices=["grouped", "leaky", "notebook", "full"])
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check: 1 epoch, few rows, results/smoke/")
    args = ap.parse_args()

    cfg = Config(**json.loads(Path(args.config).read_text()))
    rows, test_rows = load_rows("dev"), load_rows("test")
    results_dir = ROOT / "results" / ("smoke" if args.smoke else "runs")
    if args.smoke:
        cfg.epochs = 1
        test_rows = test_rows[:24]
    if args.split == "full":
        fold = np.full(len(rows), -1)
        todo = [0]
    else:
        fold = make_folds(args.split, rows, args.fold_seed)
        todo = args.folds if args.folds else fold_ids(fold)

    out_dir = results_dir / cfg.name / f"{args.split}-s{args.fold_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fold_assignment.npy", fold)

    for k in todo:
        path = out_dir / f"fold{k}.npz"
        if path.exists():
            print(f"[{cfg.name} {args.split} fold {k}] already done")
            continue
        val_idx = np.flatnonzero(fold == k) if args.split != "full" else np.array([], dtype=int)
        train_idx = np.flatnonzero(fold != k) if args.split not in ("notebook", "full") else np.flatnonzero(fold == -1)
        if args.smoke:
            train_idx, val_idx = train_idx[:48], val_idx[:24]
        print(f"[{cfg.name} {args.split} fold {k}] train {len(train_idx)} / held out {len(val_idx)}", flush=True)
        started = time.time()
        run = Run(cfg, log=lambda s: print(s, flush=True))
        run.train([rows[i] for i in train_idx])
        val_probs = run.predict([rows[i] for i in val_idx]) if len(val_idx) else np.zeros((0, run.n_heads, 3), np.float32)
        test_probs = run.predict(test_rows) if not args.no_test else np.zeros((0, run.n_heads, 3), np.float32)
        meta = {"config": cfg.to_dict(), "split": args.split, "fold_seed": args.fold_seed, "fold": k,
                "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "history": run.history,
                "train_seconds": run.train_seconds, "total_seconds": time.time() - started,
                "encoder_stats_train": run.train_stats, "device": str(run.dev), "versions": versions(),
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if len(val_idx):
            y = np.array([rows[i]["label"] for i in val_idx])
            meta["val_scores"] = scores(y, val_probs[:, 0].argmax(-1))
            print(f"[{cfg.name} {args.split} fold {k}] held-out strict F1 {meta['val_scores']['strict_f1']:.4f} "
                  f"acc {meta['val_scores']['strict_acc']:.4f}", flush=True)
        run.close()
        (out_dir / f"fold{k}.json").write_text(json.dumps(meta, indent=1))
        save_atomic(path, val_idx=val_idx, val_probs=val_probs, test_probs=test_probs)


if __name__ == "__main__":
    main()
