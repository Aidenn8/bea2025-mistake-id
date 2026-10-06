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
    # Validation scores printed by the 2025 notebook (cell 18) for the submitted model.
    out["notebook_original_reported"] = {"strict_f1": 0.6429468030944046, "strict_acc": 0.8629032258064516,
                                         "lenient_f1": 0.8406567910048188, "lenient_acc": 0.9274193548387096}
    for key, name in (("notebook_split", "ala_bert"), ("notebook_split_seed1", "ala_bert_seed1"),
                      ("notebook_split_seed2", "ala_bert_seed2")):
        if not (RUNS / name / "notebook-s0" / "fold0.npz").exists():
            continue
        nb = load_run(name, "notebook", 0)
        mask = ~np.isnan(nb["oof"][:, 0])
        pred = nb["oof"][mask].argmax(-1)
        out[key] = scores(y[mask], pred) | {"n": int(mask.sum()), "seed": nb["meta"][0]["config"]["seed"],
                                            "pred_counts": np.bincount(pred, minlength=3).tolist(),
                                            "test_pred_counts": np.bincount(nb["test"].argmax(-1), minlength=3).tolist()}
    for split in ("grouped", "leaky"):
        run = available("ala_bert", split)
        if run:
            ev = evaluate(run, y)
            ci = bootstrap(y, {"m": run["oof"].argmax(-1)}, groups, n_boot=N_BOOT)["m"]
            ev["strict_f1_ci"] = [ci["lo"], ci["hi"]]
            out[f"cv_{split}"] = ev
    return out


# ------------------------------------------------------------------ ladder and backbones
def ranking(probs, y) -> dict:
    """Threshold-free quality of the probabilities: one-vs-rest AUC and average precision per
    class, and the log-loss of the three-way distribution."""
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    out = {LABELS[c]: float(roc_auc_score(y == c, probs[:, c])) for c in range(3)}
    out["ap"] = {LABELS[c]: float(average_precision_score(y == c, probs[:, c])) for c in range(3)}
    out["log_loss"] = float(log_loss(y, probs / probs.sum(1, keepdims=True)))
    return out


def system_entry(run, y, groups, ref_pred=None):
    ev = evaluate(run, y)
    ev["auc"] = ranking(run["oof"], y)
    pred = run["oof"].argmax(-1)
    ci = bootstrap(y, {"m": pred}, groups, n_boot=N_BOOT)["m"]
    ev["strict_f1_ci"] = [ci["lo"], ci["hi"]]
    ev["confusion"] = confusion(y, pred).tolist()
    ev["minutes_per_fold"] = float(np.mean([m["total_seconds"] for m in run["meta"]]) / 60)
    if ref_pred is not None:
        ev["vs_ala"] = paired_difference(y, pred, ref_pred, groups, n_boot=N_BOOT)
    # The same model with the tuned decision rule (offsets cross-fitted over the folds).
    rule_pred, offs = crossfit_offsets(run["oof"], y, run["fold"])
    ev["rule"] = scores(y, rule_pred)
    ci = bootstrap(y, {"m": rule_pred}, groups, n_boot=N_BOOT)["m"]
    ev["rule"]["strict_f1_ci"] = [ci["lo"], ci["hi"]]
    ev["rule"]["mean_offsets"] = np.mean(offs, axis=0).tolist()
    ev["_rule_pred"] = rule_pred
    return ev


def ladder_section(y, groups):
    """Every single-model system, each with and without the decision rule. Paired differences
    against the previous ladder step and against ALA are computed with the rule applied to both
    sides, since the rule is part of the method and helps the old recipe too."""
    out, prev = {}, None
    ala = available("ala_bert")
    ref = ala["oof"].argmax(-1) if ala else None
    ala_rule = None
    for name in LADDER + [m for m in FINAL_MEMBERS if m not in LADDER]:
        run = available(name)
        if not run:
            continue
        entry = system_entry(run, y, groups, ref)
        rule_pred = entry.pop("_rule_pred")
        if name == "ala_bert":
            ala_rule = rule_pred
        elif ala_rule is not None:
            entry["rule_vs_ala_rule"] = paired_difference(y, rule_pred, ala_rule, groups, n_boot=N_BOOT)
        if prev is not None and name in LADDER:
            entry["rule_vs_previous"] = paired_difference(y, rule_pred, prev, groups, n_boot=N_BOOT)
        if name in LADDER:
            prev = rule_pred
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
    out = {"members": members, "auc": ranking(oof, y)}
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
    arrays = {"oof": oof, "pred": crossfit_pred, "test": test, "offsets": final_offsets, "fold": fold}

    # Exploratory, decided after seeing three members' results: the conversation-aware second
    # stage (mistake_id/stack.py, C = 1 as written before any result), plus a control that gets
    # only each reply's own probabilities, to separate sibling information from recalibration.
    from mistake_id.stack import crossfit_stack, features, fit_stack
    from sklearn.linear_model import LogisticRegression
    stacked = crossfit_stack(oof, y, groups, fold, C=1.0)
    st_pred, st_offsets = crossfit_offsets(stacked, y, fold)
    control = np.zeros_like(oof)
    Xc = features(oof, groups)[:, :3]
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xc[tr], y[tr])
        control[np.ix_(np.flatnonzero(te), clf.classes_)] = clf.predict_proba(Xc[te])
    ctl_pred, _ = crossfit_offsets(control, y, fold)
    ex = {"stacked": scores(y, st_pred), "stacked_auc": ranking(stacked, y),
          "control_own_only": scores(y, ctl_pred), "control_auc": ranking(control, y),
          "stacked_vs_ensemble": paired_difference(y, st_pred, crossfit_pred, groups, n_boot=N_BOOT),
          "siblings_vs_control": paired_difference(y, st_pred, ctl_pred, groups, n_boot=N_BOOT),
          "stacked_confusion": confusion(y, st_pred).tolist()}
    ci = bootstrap(y, {"s": st_pred}, groups, n_boot=N_BOOT)["s"]
    ex["stacked"]["strict_f1_ci"] = [ci["lo"], ci["hi"]]
    if ala:
        ex["stacked_vs_ala"] = paired_difference(y, st_pred, ala["oof"].argmax(-1), groups, n_boot=N_BOOT)
    stacker = fit_stack(oof, y, groups, C=1.0)
    test_groups = np.array([r["group"] for r in load_rows("test")])
    test_stacked = np.zeros_like(test)
    test_stacked[:, stacker.classes_] = stacker.predict_proba(features(test, test_groups))
    st_final_offsets = tune_offsets(stacked, y)
    ex["final_offsets"] = st_final_offsets.tolist()
    out["exploratory"] = ex
    arrays.update(stacked_pred=st_pred, stacked_test=test_stacked, stacked_offsets=st_final_offsets)
    return out, arrays


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
        if "stacked_test" in final:
            labels["final_stacked"] = apply_offsets(final["stacked_test"], final["stacked_offsets"])
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
        preds["final_stacked"] = final_arrays["stacked_pred"]
    summary["slices"] = slices_section(y, dev, preds)
    summary["test"] = test_section(test, final_arrays)
    # Scores returned by the organisers' submission server (post-shared-task phase), entered by hand
    # from the Codabench result page into results/official_scores.json.
    off = RES / "official_scores.json"
    if off.exists():
        summary["official_scores"] = json.loads(off.read_text())
        if "final" in summary["official_scores"]:
            summary["official_new"] = [summary["official_scores"]["final"]["strict_f1"], "New system"]
    (RES / "summary.json").write_text(json.dumps(round_floats(summary), indent=1))
    print(json.dumps(round_floats({k: summary[k] for k in ("ala",)}), indent=1)[:3000])
    for name, s in summary["systems"].items():
        print(f"{name:18s} strict F1 {s['pooled']['strict_f1']:.4f}  CI {s['strict_f1_ci'][0]:.3f}-{s['strict_f1_ci'][1]:.3f}"
              f"  acc {s['pooled']['strict_acc']:.4f}  lenient F1 {s['pooled']['lenient_f1']:.4f}"
              f"  AUC " + " ".join(f"{k[:4]} {s['auc'][k]:.3f}" for k in LABELS) + f"  log-loss {s['auc']['log_loss']:.4f}")
    if final:
        print("final", json.dumps(round_floats({k: final[k] for k in ("members", "ensemble_argmax", "ensemble_crossfit_offsets")}))[:1500])


if __name__ == "__main__":
    main()
