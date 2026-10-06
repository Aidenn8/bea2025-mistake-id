"""All figures for the report, from results/summary.json and the saved runs.

    python scripts/make_figures.py

Every figure is drawn at the size the page shows it (100 CSS px per inch; TEXT is the
width of a text line, FULL the width of the content column), so type reads the same size
in every figure. Black background and Inter, to sit on the report page. Series are
labelled directly where that is clearer than a legend. Colours were checked with a
colour-vision-deficiency validator against the black background: orange is the old
(submitted) system, blue the new one; the three classes use aqua, violet and red-orange.
Figures whose inputs do not exist yet are skipped.
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
for f in sorted((ROOT / "tools" / "fonts").glob("Inter-*.ttf")):
    fm.fontManager.addfont(str(f))

FULL, TEXT = 9.4, 7.6
BG, INK, DIM, FAINT, GRID, AXIS = "#000000", "#f0f0f0", "#a8a8a8", "#6e6e6e", "#1e1e1e", "#4a4a4a"
OLD, NEW = "#C28100", "#3A95D0"
CLS = {"Yes": "#199e70", "To some extent": "#9085e9", "No": "#d95926"}
NEUTRAL = "#7c7c7c"
ANN = 9.0
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "figure.constrained_layout.use": True,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "Inter", "font.size": 9.8, "axes.unicode_minus": True,
    "text.color": INK, "axes.labelcolor": DIM, "axes.labelsize": 9.6, "axes.labelpad": 6,
    "axes.titlesize": 10.3, "axes.titleweight": "medium", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.titlecolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": DIM, "ytick.labelcolor": DIM,
    "xtick.labelsize": 9.2, "ytick.labelsize": 9.2, "xtick.major.size": 3.5, "ytick.major.size": 0,
    "xtick.major.width": 0.8, "xtick.minor.size": 0, "ytick.minor.size": 0, "ytick.major.pad": 6,
    "legend.frameon": False, "legend.fontsize": 9.2, "legend.labelcolor": INK, "legend.handlelength": 1.2,
    "legend.borderaxespad": 0.2, "legend.columnspacing": 1.4,
    "lines.linewidth": 1.7, "lines.markersize": 5, "patch.linewidth": 0,
})

TUTOR_NAME = {"Expert": "Expert (human)", "Novice": "Novice (human)", "GPT4": "GPT-4", "Sonnet": "Sonnet",
              "Gemini": "Gemini", "Llama31405B": "Llama 3.1 405B", "Llama318B": "Llama 3.1 8B",
              "Mistral": "Mistral", "Phi3": "Phi-3"}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  wrote", name)


def say(ax, xy, text, color=DIM, dx=0, dy=0, ha="left", va="center", size=ANN, weight="normal", **kw):
    return ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=color,
                       fontsize=size, fontweight=weight, **kw)


def hbar_axes(ax):
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)


def load_summary():
    p = RES / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- data: label mix by tutor
def fig_label_mix(S):
    by = S["data"]["dev_by_tutor"]
    tutors = sorted(by, key=lambda t: by[t]["Yes"] / sum(by[t].values()))
    fig, ax = plt.subplots(figsize=(TEXT, 3.5))
    hbar_axes(ax)
    for i, t in enumerate(tutors):
        n = sum(by[t].values())
        left = 0.0
        for c in ("Yes", "To some extent", "No"):
            w = by[t][c] / n
            ax.barh(i, w, left=left, height=0.62, color=CLS[c], edgecolor=BG, linewidth=1.5)
            if w >= 0.07:
                say(ax, (left + w / 2, i), f"{w:.0%}", color="#ffffff", ha="center", size=8.4)
            left += w
        say(ax, (1.0, i), f"n = {n}", color=FAINT, dx=8, size=8.4)
    ax.set_yticks(range(len(tutors)), [TUTOR_NAME.get(t, t) for t in tutors])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of the tutor's responses (development set)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLS[c]) for c in CLS]
    ax.legend(handles, ["Mistake identified (Yes)", "To some extent", "Not identified (No)"], ncol=3,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Who identifies the student's mistake", pad=30)
    save(fig, "label_mix.png")


# ---------------------------------------------------------------- data: input lengths
def fig_lengths(S):
    fig, axes = plt.subplots(2, 1, figsize=(TEXT, 3.9), sharex=True)
    bins = np.arange(0, 1450, 25)
    for ax, split in zip(axes, ("dev", "test")):
        L = np.load(RES / f"lengths_{split}.npy")
        st = S["data"][f"{split}_tokens"]
        ax.hist(L[L <= 512], bins=bins, color=NEUTRAL, edgecolor=BG, linewidth=0.6)
        ax.hist(L[L > 512], bins=bins, color=OLD, edgecolor=BG, linewidth=0.6)
        ax.axvline(512, color=DIM, lw=1, ls=(0, (3, 3)))
        name = "Development set" if split == "dev" else "Test set"
        ax.set_title(f"{name}  ·  {len(L):,} responses", fontsize=9.8, pad=8)
        pct = st["over_512"] / len(L)
        say(ax, (690, ax.get_ylim()[1] * 0.78),
            f"{st['over_512']} inputs ({pct:.0%}) longer than 512 tokens:\nthe end of the input, where the tutor's\n"
            f"response sits, is cut off; {st['response_lost']} lose it entirely", color=INK, va="top", size=8.6)
        ax.set_ylabel("Responses")
    axes[1].set_xlabel("Length of the submitted system's input (BERT tokens): history, then response")
    axes[1].set_xlim(0, 1400)
    save(fig, "lengths.png")


# ---------------------------------------------------------------- leaderboard
def fig_leaderboard(S, new_score=None, new_label=None):
    lb = json.loads((RES / "leaderboard_track1.json").read_text())
    rows = lb["rows"]
    f1 = np.array([r[3] for r in rows])
    fig, ax = plt.subplots(figsize=(TEXT, 2.5))
    hbar_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    x = np.arange(1, len(rows) + 1)
    colors = [OLD if r[1] == "ALA" else "#3c3c3c" for r in rows]
    ax.bar(x, f1, width=0.72, color=colors)
    ala = next(r for r in rows if r[1] == "ALA")
    say(ax, (ala[0], ala[3]), f"ALA (ours, submitted)\n#{ala[0]} of {len(rows)}, {ala[3]:.3f}", color=INK, dy=6,
        ha="center", va="bottom", size=8.6)
    say(ax, (1, rows[0][3]), f"1st: {rows[0][3]:.3f}", color=DIM, dy=6, ha="left", va="bottom", size=8.4)
    base = lb["baseline_majority_test"]["strict_f1"]
    edge = len(rows) + 0.7
    ax.plot([0.3, edge], [base, base], color=FAINT, lw=0.9, ls=(0, (3, 3)), zorder=0)
    say(ax, (edge, base), f"Always “Yes”\n{base:.3f}", color=DIM, dx=6, ha="left", size=8.4)
    if new_score is not None:
        rank = int((f1 > new_score).sum()) + 1
        ax.plot([0.3, edge], [new_score, new_score], color=NEW, lw=1.4, zorder=3)
        say(ax, (edge, new_score), f"{new_label}\n{new_score:.3f} (#{rank})", color=INK, dx=6, ha="left", size=8.6)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, len(rows) + 9)
    ax.spines["bottom"].set_bounds(0.3, len(rows) + 0.7)
    ax.set_xticks([1, 10, 20, 30, 40, 44])
    ax.set_xlabel("Rank among the 44 teams on the official Track 1 leaderboard")
    ax.set_ylabel("Test macro-F1 (strict)")
    save(fig, "leaderboard.png")


# ---------------------------------------------------------------- how the old system scores, by protocol
def fig_protocol(S):
    A = S["ala"]
    rows = []   # (label, value, lo, hi, style)
    rows.append(("Reported in the 2025 notebook\n80/20 split of replies", A["notebook_original_reported"]["strict_f1"], None, None, "hollow"))
    for name, seed in (("notebook_split", 42), ("seed1", 1), ("seed2", 2)):
        r = A.get(name if name == "notebook_split" else f"notebook_split_{name}")
        if r:
            rows.append((f"Same recipe retrained, seed {seed}\nsame 80/20 split", r["strict_f1"], None, None, "dot"))
    for split, label in (("leaky", "Cross-validation over replies\n(leaky), 5 folds pooled"),
                         ("grouped", "Cross-validation over conversations\n5 folds pooled")):
        r = A.get(f"cv_{split}")
        if r:
            rows.append((label, r["pooled"]["strict_f1"], *r["strict_f1_ci"], "dot"))
    rows.append(("Official hidden test set", A["official_test"]["strict_f1"], None, None, "solid"))
    fig, ax = plt.subplots(figsize=(TEXT, 0.48 * len(rows) + 0.9))
    hbar_axes(ax)
    for i, (label, v, lo, hi, style) in enumerate(rows):
        y = len(rows) - 1 - i
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=OLD, lw=1.6, solid_capstyle="butt")
        face = {"hollow": BG, "dot": OLD, "solid": OLD}[style]
        ax.plot(v, y, marker="D" if style == "solid" else "o", ms=7.5, mfc=face, mec=OLD, mew=1.6, ls="none", zorder=3)
        say(ax, (max(v, hi or v), y), f"{v:.3f}", color=INK, dx=10, size=8.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8.8)
    ax.set_xlim(0.45, 0.75)
    ax.set_xlabel("Macro-F1 (strict) of the submitted system's recipe")
    save(fig, "protocol.png")


# ---------------------------------------------------------------- ablation ladder and backbones
LADDER_ROWS = [("ala_bert", "ALA, submitted recipe (BERT)"), ("bert_pair", "+ input: reply first, history trimmed"),
               ("bert_pair_opt", "+ warm-up, learning rate, class weights"), ("bert_pair_opt_mt", "+ multi-task heads"),
               ("roberta_mt", "Full recipe, RoBERTa"), ("mpnet_mt", "Full recipe, MPNet"),
               ("modernbert_mt", "Full recipe, ModernBERT")]


def fig_ladder(S):
    sysd, F = S["systems"], S.get("final")
    rows = [(lab, sysd[k]["pooled"]["strict_f1"], *sysd[k]["strict_f1_ci"], OLD if k == "ala_bert" else NEUTRAL)
            for k, lab in LADDER_ROWS if k in sysd]
    if F:
        rows.append((f"Ensemble of {len(F['members'])} backbones", F["ensemble_argmax"]["strict_f1"],
                     *F["ensemble_argmax"]["strict_f1_ci"], NEW))
        rows.append(("Ensemble + tuned decision rule", F["ensemble_crossfit_offsets"]["strict_f1"],
                     *F["ensemble_crossfit_offsets"]["strict_f1_ci"], NEW))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(TEXT, 0.42 * len(rows) + 1.0))
    hbar_axes(ax)
    tf = S["baselines"]["tfidf_response_grouped"]["strict_f1"]
    ax.axvline(tf, color=FAINT, lw=0.9, ls=(0, (3, 3)), zorder=0)
    say(ax, (tf, len(rows) - 0.4), "TF-IDF + logistic\nregression, reply only", color=DIM, dx=-4, ha="right", va="bottom", size=8.2)
    for i, (label, v, lo, hi, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt")
        ax.plot(v, y, "o", ms=7, color=col, zorder=3)
        say(ax, (hi, y), f"{v:.3f}", color=INK, dx=8, size=8.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8.8)
    ax.set_xlim(0.45, 0.8)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Macro-F1 (strict), conversation-grouped 5-fold cross-validation, with 95% interval")
    save(fig, "ladder.png")


# ---------------------------------------------------------------- old vs new, per class
def fig_per_class(S):
    sl = S["slices"]
    if "ala" not in sl or "final" not in sl:
        return
    classes = ["Yes", "To some extent", "No"]
    fig, ax = plt.subplots(figsize=(TEXT, 2.5))
    hbar_axes(ax)
    for j, (key, col, name) in enumerate((("ala", OLD, "ALA recipe"), ("final", NEW, "New system"))):
        vals = [sl[key]["per_class"][f"f1_{c}"] for c in classes]
        ys = np.arange(len(classes))[::-1] + (0.18 if j == 0 else -0.18)
        ax.barh(ys, vals, height=0.34, color=col, label=name)
        for y, v in zip(ys, vals):
            say(ax, (v, y), f"{v:.2f}", color=INK, dx=6, size=8.6)
    ax.set_yticks(np.arange(len(classes))[::-1], classes)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 per class, cross-validated")
    ax.legend(loc="lower right", ncol=1)
    save(fig, "per_class.png")


def fig_confusion(S):
    sysd, F = S["systems"], S.get("final")
    if "ala_bert" not in sysd or not F:
        return
    mats = [("ALA recipe", np.array(sysd["ala_bert"]["confusion"]), OLD), ("New system", np.array(F["confusion"]), NEW)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 3.0))
    names = ["Yes", "Some", "No"]
    for ax, (title, cm, col) in zip(axes, mats):
        ax.grid(False)
        frac = cm / cm.sum(1, keepdims=True)
        rgb = np.array(matplotlib.colors.to_rgb(col))
        img = np.array([[rgb * f + np.array([0.06, 0.06, 0.06]) * (1 - f) for f in row] for row in frac])
        ax.imshow(img, aspect="equal")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{frac[i, j]:.0%}\n{cm[i, j]}", ha="center", va="center", fontsize=8.6,
                        color="#ffffff" if frac[i, j] < 0.75 else "#000000")
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True label")
        ax.set_title(title, fontsize=9.8)
    save(fig, "confusion.png")


def fig_by_length(S):
    sl = S["slices"]
    if "ala" not in sl or "final" not in sl:
        return
    bins = list(sl["ala"]["by_length"])
    fig, ax = plt.subplots(figsize=(TEXT, 2.6))
    x = np.arange(len(bins))
    for key, col, name, dx in (("ala", OLD, "ALA recipe", -0.17), ("final", NEW, "New system", 0.17)):
        v = [sl[key]["by_length"][b]["strict_acc"] for b in bins]
        ax.bar(x + dx, v, width=0.32, color=col, label=name)
        for xi, vi in zip(x + dx, v):
            say(ax, (xi, vi), f"{vi:.0%}", color=INK, dy=4, ha="center", va="bottom", size=8.2)
    labels = [f"{b.split('-')[0]}–{b.split('-')[1]}" if b.split("-")[1] != "10000" else f"over {b.split('-')[0]}" for b in bins]
    ns = [sl["ala"]["by_length"][b]["n"] for b in bins]
    ax.set_xticks(x, [f"{lab}\n{n} replies" for lab, n in zip(labels, ns)])
    ax.set_ylim(0.5, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Input length in the submitted format (BERT tokens)")
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.16))
    save(fig, "by_length.png")


def fig_by_tutor(S):
    sl = S["slices"]
    if "ala" not in sl or "final" not in sl:
        return
    tutors = sorted(sl["ala"]["by_tutor"], key=lambda t: sl["final"]["by_tutor"][t]["strict_acc"])
    fig, ax = plt.subplots(figsize=(TEXT, 3.3))
    hbar_axes(ax)
    for i, t in enumerate(tutors):
        a, b = sl["ala"]["by_tutor"][t]["strict_acc"], sl["final"]["by_tutor"][t]["strict_acc"]
        ax.plot([a, b], [i, i], color=AXIS, lw=1.6, zorder=1)
        ax.plot(a, i, "o", ms=7, color=OLD, zorder=2)
        ax.plot(b, i, "o", ms=7, color=NEW, zorder=3)
        say(ax, (max(a, b), i), f"{a:.0%} → {b:.0%}", color=DIM, dx=9, size=8.4)
    ax.set_yticks(range(len(tutors)), [TUTOR_NAME.get(t, t) for t in tutors])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0.5, 1.08)
    ax.set_xlabel("Accuracy on the tutor's replies, cross-validated")
    h = [plt.Line2D([], [], marker="o", ls="none", color=OLD, ms=7), plt.Line2D([], [], marker="o", ls="none", color=NEW, ms=7)]
    ax.legend(h, ["ALA recipe", "New system"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "by_tutor.png")


def main():
    S = load_summary()
    if S is None:
        sys.exit("results/summary.json missing; run scripts/analyze.py first")
    fig_label_mix(S)
    fig_lengths(S)
    final_test = S.get("official_new")
    fig_leaderboard(S, *(final_test if final_test else (None, None)))
    fig_protocol(S)
    fig_ladder(S)
    fig_per_class(S)
    fig_confusion(S)
    fig_by_length(S)
    fig_by_tutor(S)


if __name__ == "__main__":
    main()
