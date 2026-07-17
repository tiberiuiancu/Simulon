"""Shared plotting utilities for experiment validation figures.

Designed for single-column LaTeX PDF export: compact sizing, serif fonts,
clean titles, and a horizontal legend placed under the title.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def setup_latex_style() -> None:
    """Configure matplotlib for full-page-width LaTeX PDF figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.3,
            "axes.titlepad": 4,
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
    """Draw a real-vs-simulated metric panel on *ax* with centered bar groups.

    Each model's bars are centered on its x-tick regardless of how many
    sources the model has.  Percentage-difference labels are placed on top
    of the *simulated* bar they refer to (not on the tick center).
    """
    palette = palette or {
        "Real": "#4c72b0",
        "Simulated": "#dd8452",
        "Simulated (no comm)": "#2ca02c",
    }

    all_sources = [s for s in palette if s in sub_df["source"].unique()]
    models = list(sub_df["model"].unique())

    bar_width = 0.28
    y_max = sub_df["value"].max()

    real_by_model: dict[str, float] = {}
    for model in models:
        real_rows = sub_df[(sub_df["model"] == model) & (sub_df["source"] == "Real")]["value"]
        if not real_rows.empty:
            real_by_model[model] = float(real_rows.iloc[0])

    legend_added: set[str] = set()
    for i, model in enumerate(models):
        model_sources = [
            s
            for s in all_sources
            if not sub_df[(sub_df["model"] == model) & (sub_df["source"] == s)]["value"].empty
        ]
        n_bars = len(model_sources)
        # symmetric offsets so the group is centered on x=i
        offsets = [(j - (n_bars - 1) / 2) * bar_width for j in range(n_bars)]
        for j, source in enumerate(model_sources):
            val = float(
                sub_df[(sub_df["model"] == model) & (sub_df["source"] == source)]["value"].iloc[0]
            )
            x_pos = i + offsets[j]
            label = source if source not in legend_added else None
            legend_added.add(source)
            ax.bar(
                x_pos,
                val,
                width=bar_width * 0.92,
                color=palette[source],
                label=label,
                edgecolor="none",
            )
            # percentage label on top of simulated bars
            if source != "Real":
                real = real_by_model.get(model)
                if real and real > 0:
                    pct = (val - real) / real * 100
                    ax.text(
                        x_pos,
                        val + 0.02 * y_max,
                        f"{pct:+.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="red",
                    )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    if legend_added:
        ax.legend(
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=max(1, len(legend_added)),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )
    sns.despine(ax=ax, top=True, right=True)


def make_figure(title: str, width_in: float = 6.5, n_panels: int = 1) -> tuple:
    """Create a figure sized for full-page-width LaTeX inclusion."""
    if n_panels <= 1:
        fig, axes = plt.subplots(1, 1, figsize=(width_in, 2.2))
        axes = [axes]
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=(width_in, 2.0 * n_panels), sharey=False)
        axes = list(axes)
    fig.suptitle(title, fontsize=8, fontweight="bold", y=0.98)
    return fig, axes
