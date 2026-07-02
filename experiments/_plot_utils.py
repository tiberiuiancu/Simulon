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
    labels: dict[str, str] = {
        "gptoss-bf16": "GPT-OSS 20B\n(EP8 PP2)",
        "gptoss-bf16-3nic": "GPT-OSS 20B\n(75% BW)",
        "gptoss-bf16-2nic": "GPT-OSS 20B\n(50% BW)",
        "gptoss-bf16-1nic": "GPT-OSS 20B\n(25% BW)",
        "gptoss-bf16-pp4-ep4": "GPT-OSS 20B\n(EP4 PP4)",
        "gptoss-fp8": "GPT-OSS 20B\n(FP8)",
        "gptoss-fp8-3nic": "GPT-OSS 20B\n(FP8, 75% BW)",
        "llama3-70b": "Llama 3\n70B",
        "qwen3-32b": "Qwen3-32B\n(TP4 PP1)",
        "qwen3-32b-tp4-pp2-mbs2-vpp8": "Qwen3-32B\n(TP4 PP2 VPP8)",
        "qwen3-32b-tp2-pp4-mbs1-vpp1": "Qwen3-32B\n(TP2 PP4)",
        "gptoss-120b": "GPT-OSS\n120B",
        "qwen3-30b": "Qwen3 30B\nA3B",
        "qwen3-30b-overlap": "Qwen3 30B\nA3B",
        "qwen3-235b": "Qwen3 235B\nA22B",
        "deepseekv3": "DeepSeek",
        "deepseekv3-overlap": "DeepSeek\n(overlap)",
    }
    return labels.get(raw_name, raw_name.replace("-", " "))


def plot_metric_panel(
    ax, sub_df, metric_label: str, ylabel: str, palette: dict[str, str] | None = None
) -> None:
    """Draw a real-vs-simulated metric panel on *ax*.

    Adds percentage-difference labels above simulated bars and places
    the legend horizontally under the panel title.
    """
    palette = palette or {
        "Real": "#4c72b0",
        "Simulated": "#dd8452",
        "Simulated (overlap)": "#2ca02c",
    }

    sns.barplot(
        data=sub_df,
        x="model",
        y="value",
        hue="source",
        hue_order=[s for s in palette if s in sub_df["source"].unique()],
        palette=palette,
        ax=ax,
    )

    y_max = sub_df["value"].max()
    real_by_model = {
        model: float(
            sub_df[(sub_df["model"] == model) & (sub_df["source"] == "Real")]["value"].iloc[0]
        )
        for model in sub_df["model"].unique()
        if not sub_df[(sub_df["model"] == model) & (sub_df["source"] == "Real")]["value"].empty
    }
    for model_name in sub_df["model"].unique():
        real = real_by_model.get(model_name)
        if real is None or real == 0:
            continue
        for source in ["Simulated", "Simulated (overlap)"]:
            sim_val = sub_df[(sub_df["model"] == model_name) & (sub_df["source"] == source)][
                "value"
            ]
            if sim_val.empty:
                continue
            sim = float(sim_val.iloc[0])
            pct = (sim - real) / real * 100
            ax.text(
                model_name,
                sim + 0.02 * y_max,
                f"{pct:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=6,
                color="red",
            )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    if not sub_df["source"].empty:
        ax.legend(
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=max(1, len(sub_df["source"].unique())),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )
    sns.despine(ax=ax, top=True, right=True)


def make_figure(title: str, width_in: float = 3.5, n_panels: int = 1) -> tuple:
    """Create a compact figure suitable for a single LaTeX column."""
    if n_panels <= 1:
        fig, axes = plt.subplots(1, 1, figsize=(width_in, 2.0))
        axes = [axes]
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=(width_in, 2.0 * n_panels), sharey=False)
        axes = list(axes)
    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.04)
    return fig, axes
