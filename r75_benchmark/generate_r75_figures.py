import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
RESULT_DIR = "results"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Shared settings
# ============================================================
METHODS = ["MultiVI", "GLUE", "MOFA+"]

COLORS = {
    "MultiVI": "#4F7DBA",
    "GLUE": "#5AAE6A",
    "MOFA+": "#C84E55",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


# ============================================================
# Helper functions
# ============================================================
def find_method_column(df):
    candidates = ["method", "Method", "model", "model_label", "Model"]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"No method column found. Available columns: {df.columns.tolist()}")


def get_ordered_values(df, value_col):
    method_col = find_method_column(df)
    values = []

    for method in METHODS:
        rows = df[df[method_col].astype(str).str.contains(method, case=False, regex=False)]

        if rows.empty:
            raise ValueError(
                f"Cannot find method '{method}' in column '{method_col}'. "
                f"Available values: {df[method_col].tolist()}"
            )

        if value_col not in df.columns:
            raise KeyError(
                f"Column '{value_col}' not found. Available columns: {df.columns.tolist()}"
            )

        values.append(float(rows.iloc[0][value_col]))

    return values


def save_figure(fig, outfile_base):
    png_path = os.path.join(FIG_DIR, f"{outfile_base}.png")
    pdf_path = os.path.join(FIG_DIR, f"{outfile_base}.pdf")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_simple_bar(values, title, ylabel, outfile_base, ylim=None, lower_is_better=False):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    bars = ax.bar(
        METHODS,
        values,
        color=[COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.9,
        width=0.65,
    )

    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel)

    if ylim is None:
        ymax = max(values) * 1.25 if max(values) > 0 else 1.0
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if lower_is_better:
        ax.text(
            0.98,
            0.95,
            "Lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    plt.tight_layout()
    save_figure(fig, outfile_base)


def plot_grouped_benchmark(df, metric_map, title, ylabel, outfile_base, ylim=(0, 1.10)):
    plot_rows = []

    for display_name, col in metric_map.items():
        values = get_ordered_values(df, col)

        for method, value in zip(METHODS, values):
            plot_rows.append({
                "Metric": display_name,
                "Method": method,
                "Value": value,
            })

    plot_df = pd.DataFrame(plot_rows)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    x = np.arange(len(metric_map))
    width = 0.23

    for i, method in enumerate(METHODS):
        values = plot_df[plot_df["Method"] == method]["Value"].values

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=method,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.8,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(list(metric_map.keys()))
    ax.set_ylim(*ylim)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    save_figure(fig, outfile_base)


# ============================================================
# 1. r75 cell-type conservation
# ============================================================
def plot_r75_celltype_conservation():
    input_csv = os.path.join(RESULT_DIR, "strict_r75_benchmark_metrics.csv")
    df = pd.read_csv(input_csv)

    metric_map = {
        "NMI": "NMI",
        "Balanced silhouette": "Silhouette_balanced_celltype",
        "Balanced KNN purity": "KNN_purity_balanced_celltype",
    }

    plot_grouped_benchmark(
        df=df,
        metric_map=metric_map,
        title="r75 cell-type conservation benchmark",
        ylabel="Metric value",
        outfile_base="figure2G_r75_celltype_conservation_benchmark",
        ylim=(0, 1.10),
    )


# ============================================================
# 2. r75 ARI
# ============================================================
def plot_r75_ari():
    input_csv = os.path.join(RESULT_DIR, "r75_ari_summary.csv")
    df = pd.read_csv(input_csv)

    values = get_ordered_values(df, "ARI")

    plot_simple_bar(
        values=values,
        title="r75 ARI",
        ylabel="ARI",
        outfile_base="supp_r75_ari",
        ylim=(0, 1.05),
    )


# ============================================================
# 3. r75 immune programme smoothness
# ============================================================
def plot_r75_immune_smoothness():
    input_csv = os.path.join(RESULT_DIR, "r75_immune_programme_smoothness_metrics.csv")
    df = pd.read_csv(input_csv)

    values = get_ordered_values(df, "overall_immune_programme_smoothness")

    plot_simple_bar(
        values=values,
        title="r75 immune programme smoothness",
        ylabel="Immune smoothness",
        outfile_base="supp_r75_immune_programme_smoothness",
        ylim=None,
        lower_is_better=True,
    )


# ============================================================
# 4. r75 cross-modal robustness
# ============================================================
def plot_r75_cross_modal_robustness():
    metric_csv = os.path.join(RESULT_DIR, "strict_r75_benchmark_metrics.csv")
    direct_csv = os.path.join(RESULT_DIR, "r75_direct_rna_to_atac_agreement_K15_summary.csv")

    metric_df = pd.read_csv(metric_csv)
    direct_df = pd.read_csv(direct_csv)

    rows = []

    for method in METHODS:
        metric_method_col = find_method_column(metric_df)
        direct_method_col = find_method_column(direct_df)

        metric_row = metric_df[
            metric_df[metric_method_col].astype(str).str.contains(method, case=False, regex=False)
        ]

        direct_row = direct_df[
            direct_df[direct_method_col].astype(str).str.contains(method, case=False, regex=False)
        ]

        if metric_row.empty:
            raise ValueError(f"Cannot find {method} in {metric_csv}")

        if direct_row.empty:
            raise ValueError(f"Cannot find {method} in {direct_csv}")

        rows.append({
            "Metric": "Cross-modal\nKNN purity",
            "Method": method,
            "Value": float(metric_row.iloc[0]["Cross_modal_KNN_purity"]),
        })

        rows.append({
            "Metric": "Cross-modal\nagreement",
            "Method": method,
            "Value": float(metric_row.iloc[0]["Cross_modal_celltype_agreement"]),
        })

        rows.append({
            "Metric": "Direct RNA-to-ATAC\nagreement",
            "Method": method,
            "Value": float(direct_row.iloc[0]["balanced_direct_rna_to_atac_agreement"]),
        })

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.8, 4.3))

    metrics = ["Cross-modal\nKNN purity", "Cross-modal\nagreement", "Direct RNA-to-ATAC\nagreement"]
    x = np.arange(len(metrics))
    width = 0.23

    for i, method in enumerate(METHODS):
        values = plot_df[plot_df["Method"] == method]["Value"].values

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=method,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.8,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("r75 cross-modal robustness", fontweight="bold", pad=10)
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.10)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    save_figure(fig, "supp_r75_cross_modal_robustness")


# ============================================================
# 5. r75 modality entropy
# ============================================================
def plot_r75_modality_entropy():
    input_csv = os.path.join(RESULT_DIR, "strict_r75_benchmark_metrics.csv")
    df = pd.read_csv(input_csv)

    values = get_ordered_values(df, "KNN_modality_entropy")

    plot_simple_bar(
        values=values,
        title="r75 modality mixing",
        ylabel="KNN modality entropy",
        outfile_base="supp_r75_modality_entropy",
        ylim=(0, 1.05),
    )


# ============================================================
# 6. r75 same-modality enrichment
# ============================================================
def plot_r75_same_modality_enrichment():
    input_csv = os.path.join(RESULT_DIR, "strict_r75_benchmark_metrics.csv")
    df = pd.read_csv(input_csv)

    values = get_ordered_values(df, "same_modality_enrichment")

    ymax = max(values) * 1.25 if max(values) > 0 else 3.0

    plot_simple_bar(
        values=values,
        title="r75 same-modality enrichment",
        ylabel="Same-modality enrichment",
        outfile_base="supp_r75_same_modality_enrichment",
        ylim=(0, ymax),
        lower_is_better=True,
    )


# ============================================================
# 7. r75 direct RNA-to-ATAC agreement alone
# ============================================================
def plot_r75_direct_rna_to_atac():
    input_csv = os.path.join(RESULT_DIR, "r75_direct_rna_to_atac_agreement_K15_summary.csv")
    df = pd.read_csv(input_csv)

    values = get_ordered_values(df, "balanced_direct_rna_to_atac_agreement")

    plot_simple_bar(
        values=values,
        title="r75 direct RNA-to-ATAC agreement",
        ylabel="Balanced agreement",
        outfile_base="supp_r75_direct_rna_to_atac_agreement",
        ylim=(0, 1.05),
    )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating r75 quantitative benchmark figures...")

    plot_r75_celltype_conservation()
    plot_r75_ari()
    plot_r75_immune_smoothness()
    plot_r75_cross_modal_robustness()
    plot_r75_modality_entropy()
    plot_r75_same_modality_enrichment()
    plot_r75_direct_rna_to_atac()

    print("Done. All r75 figures saved to figures/")
