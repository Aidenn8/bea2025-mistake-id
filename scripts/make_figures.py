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


def main():
    S = load_summary()
    if S is None:
        sys.exit("results/summary.json missing; run scripts/analyze.py first")
    fig_label_mix(S)
    fig_lengths(S)
    fig_leaderboard(S)


if __name__ == "__main__":
    main()
