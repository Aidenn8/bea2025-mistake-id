"""Compute every number reported on the results page from the saved runs.

    python scripts/analyze.py

Reads results/runs/, results/baselines.json and results/leaderboard_track1.json and writes
results/summary.json plus the test-set label arrays in results/test_predictions/. Runs
that have not finished are skipped, so this can be re-run while experiments are queued.

The final system was fixed before any neural result was seen: the mean of the class
probabilities of every full-recipe backbone that completed (FINAL_MEMBERS), followed by
decision offsets tuned on out-of-fold predictions (cross-fitted for the reported score).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mistake_id.data import LABELS, load_rows  # noqa: E402
from mistake_id.encode import Encoder, notebook_text  # noqa: E402
from mistake_id.metrics import bootstrap, confusion, paired_difference, scores  # noqa: E402
from mistake_id.results import RUNS, apply_offsets, average, crossfit_offsets, evaluate, load_run, tune_offsets  # noqa: E402

RES = ROOT / "results"
LADDER = ["ala_bert", "bert_pair", "bert_pair_opt", "bert_pair_opt_mt"]
FINAL_MEMBERS = ["bert_pair_opt_mt", "roberta_mt", "mpnet_mt", "modernbert_mt"]
NAMES = {
    "ala_bert": "ALA (submitted system)",
    "bert_pair": "+ input fix",
    "bert_pair_opt": "+ training recipe",
    "bert_pair_opt_mt": "+ multi-task",
    "roberta_mt": "RoBERTa-base",
    "mpnet_mt": "MPNet",
    "modernbert_mt": "ModernBERT-base",
}
MATHDIAL_PREFIX = "tutor: hi, could you please provide a step-by-step solution"
N_BOOT = 2000


def available(name, split="grouped", seed=0):
    d = RUNS / name / f"{split}-s{seed}"
    if not (d / "fold_assignment.npy").exists():
        return None
    run = load_run(name, split, seed)
    return run if run["complete"] else None


def round_floats(obj, nd=5):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, nd) for v in obj]
    if isinstance(obj, np.generic):
        return round_floats(obj.item(), nd)
    return obj


# ------------------------------------------------------------------ data
def data_section(dev, test):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    out = {"n_dev": len(dev), "n_test": len(test),
           "n_dev_conversations": len({r["conversation_id"] for r in dev}),
           "n_dev_histories": len({r["group"] for r in dev}),
           "n_test_conversations": len({r["group"] for r in test})}
    y = [r["label"] for r in dev]
    out["dev_label_counts"] = {LABELS[k]: v for k, v in sorted(Counter(y).items())}
    # The hidden test labels' class mix follows exactly from the organisers' majority-class
    # baseline (accuracy 0.7363 strict, 0.8255 lenient on 1,547 responses).
    lb = json.loads((RES / "leaderboard_track1.json").read_text())["baseline_majority_test"]
    yes = round(lb["strict_acc"] * len(test))
    yes_some = round(lb["lenient_acc"] * len(test))
    assert abs(yes / len(test) - lb["strict_acc"]) < 5e-5 and abs(yes_some / len(test) - lb["lenient_acc"]) < 5e-5
    out["test_label_counts_derived"] = {"Yes": yes, "To some extent": yes_some - yes, "No": len(test) - yes_some}

    by_tutor = {}
    for r in dev:
        by_tutor.setdefault(r["tutor"], Counter())[r["label"]] += 1
    out["dev_by_tutor"] = {t: {LABELS[k]: c[k] for k in range(3)} for t, c in by_tutor.items()}
    src = Counter("MathDial" if r["history_key"].startswith(MATHDIAL_PREFIX) else "Bridge" for r in dev if r["tutor"] == "Expert")
    out["dev_sources"] = dict(src)

    def lengths(rows):
        full = np.array([len(tok(notebook_text(r))["input_ids"]) for r in rows])
        tail = np.array([len(tok("Tutor: " + r["response"], add_special_tokens=False)["input_ids"]) for r in rows])
        head = full - 2 - tail
        cut = full > 512
        gone = cut & (510 - head <= 2)
        return full, cut, gone

    for name, rows in (("dev", dev), ("test", test)):
        full, cut, gone = lengths(rows)
        out[f"{name}_tokens"] = {"median": float(np.median(full)), "p90": float(np.percentile(full, 90)),
                                 "max": int(full.max()), "over_512": int(cut.sum()), "response_lost": int(gone.sum())}
        np.save(RES / f"lengths_{name}.npy", full)
        out[f"{name}_cut_mask_count"] = int(cut.sum())
    enc = Encoder(tok, "pair", 512, 96)
    for r in test:
        enc(r)
    out["test_pair_history_cut"] = enc.stats["history_cut"]
    return out


# ------------------------------------------------------------------ old system
def ala_section(y, groups, official):
    out = {"official_test": official}
    nb = load_run("ala_bert", "notebook", 0) if (RUNS / "ala_bert" / "notebook-s0" / "fold0.npz").exists() else None
    if nb:
        mask = ~np.isnan(nb["oof"][:, 0])
        out["notebook_split"] = scores(y[mask], nb["oof"][mask].argmax(-1))
        out["notebook_split"]["n"] = int(mask.sum())
        out["notebook_original_reported"] = {"strict_f1": 0.6429468030944046, "strict_acc": 0.8629032258064516,
                                             "lenient_f1": 0.8406567910048188, "lenient_acc": 0.9274193548387096}
    for split in ("grouped", "leaky"):
        run = available("ala_bert", split)
        if run:
            ev = evaluate(run, y)
            ci = bootstrap(y, {"m": run["oof"].argmax(-1)}, groups, n_boot=N_BOOT)["m"]
            ev["strict_f1_ci"] = [ci["lo"], ci["hi"]]
            out[f"cv_{split}"] = ev
    return out


# ------------------------------------------------------------------ ladder and backbones
def system_entry(run, y, groups, ref_pred=None):
    ev = evaluate(run, y)
    pred = run["oof"].argmax(-1)
    ci = bootstrap(y, {"m": pred}, groups, n_boot=N_BOOT)["m"]
    ev["strict_f1_ci"] = [ci["lo"], ci["hi"]]
    ev["confusion"] = confusion(y, pred).tolist()
    ev["minutes_per_fold"] = float(np.mean([m["total_seconds"] for m in run["meta"]]) / 60)
    if ref_pred is not None:
        ev["vs_ala"] = paired_difference(y, pred, ref_pred, groups, n_boot=N_BOOT)
    return ev


def ladder_section(y, groups):
    out, prev = {}, None
    ala = available("ala_bert")
    ref = ala["oof"].argmax(-1) if ala else None
    for name in LADDER + [m for m in FINAL_MEMBERS if m not in LADDER]:
        run = available(name)
        if not run:
            continue
        entry = system_entry(run, y, groups, ref)
        if prev is not None and name in LADDER:
            entry["vs_previous"] = paired_difference(y, run["oof"].argmax(-1), prev, groups, n_boot=N_BOOT)
        if name in LADDER:
            prev = run["oof"].argmax(-1)
        entry["label"] = NAMES.get(name, name)
        out[name] = entry
    return out


# ------------------------------------------------------------------ final system
def final_section(y, groups, dev):
    runs = {m: available(m) for m in FINAL_MEMBERS}
    members = [m for m, r in runs.items() if r]
    if not members:
        return None, None
    fold = runs[members[0]]["fold"]
    assert all(np.array_equal(runs[m]["fold"], fold) for m in members)
    oof = average([runs[m]["oof"] for m in members])
    out = {"members": members}
    plain = oof.argmax(-1)
    out["ensemble_argmax"] = scores(y, plain)
    crossfit_pred, chosen = crossfit_offsets(oof, y, fold)
    out["ensemble_crossfit_offsets"] = scores(y, crossfit_pred)
    out["crossfit_offsets_by_fold"] = chosen
    final_offsets = tune_offsets(oof, y)
    out["final_offsets"] = final_offsets.tolist()
    out["final_offsets_in_sample"] = scores(y, apply_offsets(oof, final_offsets))
    ci = bootstrap(y, {"argmax": plain, "crossfit": crossfit_pred}, groups, n_boot=N_BOOT)
    out["ensemble_argmax"]["strict_f1_ci"] = [ci["argmax"]["lo"], ci["argmax"]["hi"]]
    out["ensemble_crossfit_offsets"]["strict_f1_ci"] = [ci["crossfit"]["lo"], ci["crossfit"]["hi"]]
    out["confusion"] = confusion(y, crossfit_pred).tolist()
    ala = available("ala_bert")
    if ala:
        out["vs_ala"] = paired_difference(y, crossfit_pred, ala["oof"].argmax(-1), groups, n_boot=N_BOOT)
        out["vs_ala_lenient"] = paired_difference(y, crossfit_pred, ala["oof"].argmax(-1), groups, "lenient_f1", N_BOOT)
    best_single = max(members, key=lambda m: scores(y, runs[m]["oof"].argmax(-1))["strict_f1"])
    out["best_single"] = best_single
    out["vs_best_single"] = paired_difference(y, plain, runs[best_single]["oof"].argmax(-1), groups, n_boot=N_BOOT)
    # leave-one-member-out
    if len(members) > 2:
        out["leave_one_out"] = {m: scores(y, average([runs[o]["oof"] for o in members if o != m]).argmax(-1))["strict_f1"]
                                for m in members}
    test = average([runs[m]["test"] for m in members])
    return out, {"oof": oof, "pred": crossfit_pred, "test": test, "offsets": final_offsets, "fold": fold}


# ------------------------------------------------------------------ slices
def slices_section(y, dev, preds: dict):
    out = {}
    tutors = sorted({r["tutor"] for r in dev})
    tut = np.array([r["tutor"] for r in dev])
    lengths = np.load(RES / "lengths_dev.npy")
    bins = [(0, 256), (256, 384), (384, 512), (512, 10_000)]
    for name, pred in preds.items():
        s = {"by_tutor": {t: scores(y[tut == t], pred[tut == t]) | {"n": int((tut == t).sum())} for t in tutors},
             "by_length": {f"{lo}-{hi}": scores(y[(lengths >= lo) & (lengths < hi)], pred[(lengths >= lo) & (lengths < hi)])
                           | {"n": int(((lengths >= lo) & (lengths < hi)).sum())} for lo, hi in bins},
             "per_class": {k: v for k, v in scores(y, pred).items() if k.startswith("f1_")}}
        cut = lengths > 512
        s["cut_rows"] = {"n": int(cut.sum()), "acc": float((pred[cut] == y[cut]).mean()),
                         "acc_uncut": float((pred[~cut] == y[~cut]).mean())}
        out[name] = s
    return out


# ------------------------------------------------------------------ test-set predictions
def test_section(test, final):
    out = {}
    pred_dir = RES / "test_predictions"
    pred_dir.mkdir(exist_ok=True)
    lengths = np.load(RES / "lengths_test.npy")
    cut = lengths > 512
    labels = {}
    nb = RUNS / "ala_bert" / "notebook-s0" / "fold0.npz"
    if nb.exists():
        labels["ala_replica"] = np.load(nb)["test_probs"][:, 0].argmax(-1)
    if final is not None:
        labels["final"] = apply_offsets(final["test"], final["offsets"])
        labels["final_argmax"] = final["test"].argmax(-1)
    for name, lab in labels.items():
        np.save(pred_dir / f"{name}.npy", lab)
        out[name] = {"counts": {LABELS[k]: int((lab == k).sum()) for k in range(3)},
                     "cut_rows_counts": {LABELS[k]: int((lab[cut] == k).sum()) for k in range(3)}}
    if "ala_replica" in labels and "final" in labels:
        a, b = labels["ala_replica"], labels["final"]
        out["agreement"] = float((a == b).mean())
        out["agreement_cut_rows"] = float((a[cut] == b[cut]).mean())
    out["n_cut_rows"] = int(cut.sum())
    return out


def main():
    dev, test = load_rows("dev"), load_rows("test")
    y = np.array([r["label"] for r in dev])
    groups = np.array([r["group"] for r in dev])
    lb = json.loads((RES / "leaderboard_track1.json").read_text())
    ala_row = next(r for r in lb["rows"] if r[1] == "ALA")
    official = dict(zip(["strict_f1", "strict_acc", "lenient_f1", "lenient_acc"], ala_row[3:]))
    official["rank"], official["teams"] = ala_row[0], len(lb["rows"])

    summary = {"data": data_section(dev, test)}
    summary["baselines"] = json.loads((RES / "baselines.json").read_text())
    summary["ala"] = ala_section(y, groups, official)
    summary["systems"] = ladder_section(y, groups)
    final, final_arrays = final_section(y, groups, dev)
    summary["final"] = final
    preds = {}
    if available("ala_bert"):
        preds["ala"] = available("ala_bert")["oof"].argmax(-1)
    if final_arrays:
        preds["final"] = final_arrays["pred"]
    summary["slices"] = slices_section(y, dev, preds)
    summary["test"] = test_section(test, final_arrays)
    (RES / "summary.json").write_text(json.dumps(round_floats(summary), indent=1))
    print(json.dumps(round_floats({k: summary[k] for k in ("ala",)}), indent=1)[:3000])
    for name, s in summary["systems"].items():
        print(f"{name:18s} strict F1 {s['pooled']['strict_f1']:.4f}  CI {s['strict_f1_ci'][0]:.3f}-{s['strict_f1_ci'][1]:.3f}"
              f"  acc {s['pooled']['strict_acc']:.4f}  lenient F1 {s['pooled']['lenient_f1']:.4f}")
    if final:
        print("final", json.dumps(round_floats({k: final[k] for k in ("members", "ensemble_argmax", "ensemble_crossfit_offsets")}))[:1500])


if __name__ == "__main__":
    main()
