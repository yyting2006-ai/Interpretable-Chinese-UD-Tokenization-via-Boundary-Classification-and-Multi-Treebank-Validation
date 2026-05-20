from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "results" / "multitreebank_downstream_summary.csv"
DELTAS = ROOT / "results" / "paper_key_deltas.csv"
ABLATION = ROOT / "results" / "ablation_summary.csv"
ERRORS = ROOT / "results" / "boundary_error_summary.csv"


DATASET_ORDER = [
    "UD_Chinese-GSD",
    "UD_Chinese-GSDSimp",
    "UD_Chinese-PUD",
    "UD_Chinese-HK",
    "UD_Chinese-CFL",
]
DATASET_SHORT = {
    "UD_Chinese-GSD": "GSD",
    "UD_Chinese-GSDSimp": "GSDSimp",
    "UD_Chinese-PUD": "PUD",
    "UD_Chinese-HK": "HK",
    "UD_Chinese-CFL": "CFL",
}
SYSTEMS = [
    ("raw_char", "Raw char"),
    ("traditional_joint_lexicon", "Traditional"),
    ("single_gsd", "UD Boundary-GSD"),
    ("joint_gsd_gsdsimp", "UD Boundary-Joint"),
]
SYSTEM_COLORS = {
    "raw_char": "#B8B8B8",
    "traditional_joint_lexicon": "#7CA3C9",
    "single_gsd": "#E5A75A",
    "joint_gsd_gsdsimp": "#4F8F6B",
}
BLUE = "#5B7DB1"
ORANGE = "#D17A65"
GREEN = "#4F8F6B"
RED = "#C96F6F"
DARK = "#262626"
GRID = "#E8E8E8"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)


def save_all(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def letter(ax: plt.Axes, text: str) -> None:
    ax.text(-0.08, 1.06, text, transform=ax.transAxes, fontsize=10, weight="bold", va="top")


def metric_matrix(summary: pd.DataFrame, metric: str) -> np.ndarray:
    mat = []
    for ds in DATASET_ORDER:
        row = []
        for mode, _ in SYSTEMS:
            value = summary[(summary["dataset"] == ds) & (summary["mode"] == mode)][metric].iloc[0]
            row.append(value)
        mat.append(row)
    return np.array(mat)


def fig5_1(summary: pd.DataFrame) -> None:
    boundary = metric_matrix(summary, "strict_boundary_f1")
    span = metric_matrix(summary, "strict_span_f1")
    labels = [DATASET_SHORT[x] for x in DATASET_ORDER]
    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, mat, title in zip(
        axes,
        [boundary, span],
        ["Strict Boundary-F1", "Strict Span-F1"],
    ):
        for j, (mode, label) in enumerate(SYSTEMS):
            offset = (j - 1.5) * width
            ax.bar(
                x + offset,
                mat[:, j],
                width=width,
                color=SYSTEM_COLORS[mode],
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
        ax.set_title(title, loc="left", pad=8, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0.34, 1.0)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_ylabel("F1 score")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.05, 1.24),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.1,
    )
    letter(axes[0], "a")
    letter(axes[1], "b")
    fig.suptitle("Overall tokenization performance", x=0.02, y=1.04, ha="left", fontsize=11, weight="bold")
    fig.subplots_adjust(top=0.78, wspace=0.18)
    save_all(fig, "fig5_1_overall_tokenization_performance")


def fig5_2(deltas: pd.DataFrame) -> None:
    deltas = deltas.set_index("dataset").loc[DATASET_ORDER].reset_index()
    labels = [DATASET_SHORT[x] for x in deltas["dataset"]]
    x = np.arange(len(labels))
    width = 0.34
    b_gain = deltas["joint_vs_single_gsd_boundary"].to_numpy()
    s_gain = deltas["joint_vs_single_gsd_span"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.bar(x - width / 2, b_gain, width=width, color=BLUE, edgecolor="white", linewidth=0.5, label="Boundary-F1")
    ax.bar(x + width / 2, s_gain, width=width, color=GREEN, edgecolor="white", linewidth=0.5, label="Span-F1")
    for xi, val in zip(x - width / 2, b_gain):
        ax.text(xi, val + (0.008 if val >= 0 else -0.011), f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
    for xi, val in zip(x + width / 2, s_gain):
        ax.text(xi, val + (0.008 if val >= 0 else -0.011), f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gain over UD Boundary-GSD")
    ax.set_title("Effect of multi-treebank joint training", loc="left", pad=8, weight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.025, 0.205)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    save_all(fig, "fig5_2_joint_training_gains")


def fig5_3(deltas: pd.DataFrame) -> None:
    cols = [
        ("vs_traditional_upos_model", "UPOS\nmodel"),
        ("vs_traditional_deprel_model", "deprel\nmodel"),
        ("vs_traditional_upos_oracle", "UPOS\noracle"),
        ("vs_traditional_deprel_oracle", "deprel\noracle"),
    ]
    deltas = deltas.set_index("dataset").loc[DATASET_ORDER].reset_index()
    data = deltas[[c for c, _ in cols]].to_numpy()
    labels_y = [DATASET_SHORT[x] for x in deltas["dataset"]]
    labels_x = [l for _, l in cols]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.08, vmax=0.19, aspect="auto")
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x)
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    ax.tick_params(length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"+{data[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color="#102A43")
    ax.set_title("Downstream gains over traditional segmentation", loc="left", pad=8, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("F1 gain")
    cbar.ax.tick_params(labelsize=7)
    save_all(fig, "fig5_3_downstream_gains_heatmap")


def fig5_4(deltas: pd.DataFrame) -> None:
    deltas = deltas.set_index("dataset").loc[DATASET_ORDER].reset_index()
    labels = [DATASET_SHORT[x] for x in deltas["dataset"]]
    y = np.arange(len(labels))
    joint = deltas["joint_boundary_f1"].to_numpy()
    external = deltas["best_external_boundary"].to_numpy()
    best_labels = deltas["best_external"].str.replace("stanza_zh_gsdsimp", "Stanza").str.replace("pkuseg", "pkuseg").to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for yi, x1, x2 in zip(y, joint, external):
        ax.plot([x1, x2], [yi, yi], color="#B5B5B5", linewidth=1.2, zorder=1)
    ax.scatter(joint, y, s=42, color=GREEN, label="UD Boundary-Joint", zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(external, y, s=42, color=ORANGE, label="Best external", zorder=3, edgecolor="white", linewidth=0.6)
    for yi, x2, lab in zip(y, external, best_labels):
        ax.text(x2 + 0.004, yi, lab, va="center", fontsize=7, color="#555555")
    for yi, diff in zip(y, deltas["vs_best_external_boundary"].to_numpy()):
        xpos = max(joint[yi], external[yi]) + 0.035
        ax.text(xpos, yi, f"{diff:+.3f}", va="center", ha="left", fontsize=7.3, color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.88, 0.99)
    ax.set_xlabel("Strict Boundary-F1")
    ax.set_title("Comparison with mainstream segmenters", loc="left", pad=8, weight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    save_all(fig, "fig5_4_external_segmenter_comparison")


FEATURE_LABELS = {
    "lexicon_edge": "Lexicon\nedge",
    "char_identity": "Character\nidentity",
    "traditional_signal": "Traditional\nsignal",
    "lexicon_crossing": "Lexicon\ncrossing",
    "local_context": "Local\ncontext",
    "char_type": "Character\ntype",
    "position": "Position",
}


def fig5_5() -> None:
    df = pd.read_csv(ABLATION)
    datasets = ["UD_Chinese-GSD", "UD_Chinese-PUD", "UD_Chinese-HK"]
    order = [
        "lexicon_edge",
        "char_identity",
        "traditional_signal",
        "lexicon_crossing",
        "local_context",
        "char_type",
        "position",
    ]
    full = df[df["setting"] == "full"].set_index("dataset")["strict_boundary_f1"]
    ab = df[df["setting"] != "full"].copy()
    ab["feature"] = ab["setting"].str.replace("w/o ", "", regex=False)
    ab["drop"] = ab.apply(lambda r: full.loc[r["dataset"]] - r["strict_boundary_f1"], axis=1)
    mat = (
        ab.pivot_table(index="dataset", columns="feature", values="drop")
        .reindex(datasets)
        .reindex(columns=order)
    )
    data = mat.to_numpy()
    vmax = max(abs(data.min()), abs(data.max()), 0.026)

    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([FEATURE_LABELS[x] for x in order])
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([DATASET_SHORT[x] for x in datasets])
    ax.tick_params(length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if abs(val) > vmax * 0.45 else "#222222"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7.2, color=color)
    ax.set_title("Feature ablation: Boundary-F1 drop", loc="left", pad=8, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Full - ablated")
    cbar.ax.tick_params(labelsize=7)
    fig.text(0.015, 0.005, "Positive values indicate performance loss after feature removal.", fontsize=7, color="#555555")
    fig.subplots_adjust(bottom=0.25)
    save_all(fig, "fig5_5_feature_ablation_heatmap")


ERROR_LABELS = {
    "traditional-boundary": "Traditional-boundary",
    "other-context": "Contextual ambiguity",
    "inside-lexicon-item": "Inside lexicon item",
    "punctuation-adjacent": "Punctuation-adjacent",
    "latin-or-symbol": "Latin or symbol",
    "numeric-expression": "Numeric expression",
}


def fig5_6() -> None:
    df = pd.read_csv(ERRORS)
    overall = (
        df.groupby("error_type")[["false_positive", "false_negative", "total"]]
        .sum()
        .sort_values("total", ascending=True)
    )
    labels = [ERROR_LABELS[x] for x in overall.index]
    fp = overall["false_positive"].to_numpy()
    fn = overall["false_negative"].to_numpy()
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.5, 3.25))
    ax.barh(y, fp, color=BLUE, height=0.62, label="False positive")
    ax.barh(y, fn, left=fp, color=ORANGE, height=0.62, label="False negative")
    totals = overall["total"].to_numpy()
    for yi, total in zip(y, totals):
        ax.text(total + max(totals) * 0.012, yi, f"{int(total)}", va="center", fontsize=7.5, color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of boundary errors")
    ax.set_title("Distribution of boundary error types", loc="left", pad=8, weight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(totals) * 1.15)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    save_all(fig, "fig5_6_boundary_error_distribution")


def contact_sheet() -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    files = [
        "fig5_1_overall_tokenization_performance.png",
        "fig5_2_joint_training_gains.png",
        "fig5_3_downstream_gains_heatmap.png",
        "fig5_4_external_segmenter_comparison.png",
        "fig5_5_feature_ablation_heatmap.png",
        "fig5_6_boundary_error_distribution.png",
    ]
    thumbs = []
    for f in files:
        img = Image.open(OUT / f).convert("RGB")
        img.thumbnail((900, 420), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (940, 470), "white")
        canvas.paste(img, ((940 - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 440), f.replace(".png", ""), fill=(40, 40, 40))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1880, 1410), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % 2) * 940
        y = (idx // 2) * 470
        sheet.paste(thumb, (x, y))
    sheet.save(OUT / "chapter5_contact_sheet.png", quality=95)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    deltas = pd.read_csv(DELTAS)
    fig5_1(summary)
    fig5_2(deltas)
    fig5_3(deltas)
    fig5_4(deltas)
    fig5_5()
    fig5_6()
    contact_sheet()
    print(f"Wrote Chapter 5 figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
