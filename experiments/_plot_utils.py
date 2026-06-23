"""Shared plotting utilities for experiment validation figures.

Designed for single-column LaTeX PDF export: compact sizing, serif fonts,
clean titles, and a horizontal legend placed under the title.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def setup_latex_style() -> None:
    """Configure matplotlib for single-column LaTeX PDF figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlepad": 8,
        }
    )


def label_for_model(raw_name: str) -> str:
    """Return a presentation-friendly model label from a directory name."""
    labels: dict[str, str] = {
        "gptoss-bf16": "GPT-OSS 5B",
        "gptoss-bf16-3nic": "GPT-OSS 5B\n(3 NIC)",
        "gptoss-bf16-pp4-ep4": "GPT-OSS 5B\n(PP4 EP4)",
        "gptoss-fp8": "GPT-OSS\nFP8",
        "gptoss-fp8-3nic": "GPT-OSS FP8\n(3 NIC)",
        "llama3-70b": "Llama 3\n70B",
        "qwen3-32b": "Qwen3\n32B",
        "gptoss-120b": "GPT-OSS\n120B",
        "qwen3-30b": "Qwen3 30B\nA3B",
        "qwen3-235b": "Qwen3 235B\nA22B",
    }
    return labels.get(raw_name, raw_name.replace("-", " "))


def plot_metric_panel(
    ax, sub_df, metric_label: str, ylabel: str, palette: dict[str, str] | None = None
) -> None:
    """Draw a single real-vs-simulated metric panel on *ax*.

    Adds red percentage-difference labels above simulated bars and places
    the legend horizontally under the panel title.
    """
    palette = palette or {"Real": "#4c72b0", "Simulated": "#dd8452"}

    sns.barplot(
        data=sub_df,
        x="model",
        y="value",
        hue="source",
        hue_order=["Real", "Simulated"],
        palette=palette,
        ax=ax,
    )

    y_max = sub_df["value"].max()
    for model_name in sub_df["model"].unique():
        real_val = sub_df[(sub_df["model"] == model_name) & (sub_df["source"] == "Real")]["value"]
        sim_val = sub_df[(sub_df["model"] == model_name) & (sub_df["source"] == "Simulated")][
            "value"
        ]
        if real_val.empty or sim_val.empty:
            continue
        real = float(real_val.iloc[0])
        sim = float(sim_val.iloc[0])
        if real == 0:
            continue
        pct = (sim - real) / real * 100
        ax.text(
            model_name,
            sim + 0.02 * y_max,
            f"{pct:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    sns.despine(ax=ax, top=True, right=True)


def make_figure(title: str, width_in: float = 3.5, n_panels: int = 1) -> tuple:
    """Create a compact figure suitable for a single LaTeX column."""
    height_in = 2.0
    fig, axes = plt.subplots(1, n_panels, figsize=(width_in, height_in), sharey=False)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.04)
    return fig, axes
