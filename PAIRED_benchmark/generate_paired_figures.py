import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Paths
# ============================================================
RESULT_DIR = "results"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Shared plotting settings
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
    """
    Find method/model column from common possible column names.
    """
    candidates = ["method", "Method", "model", "model_label", "Model"]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"No method column found. Available columns: {df.columns.tolist()}"
    )


def get_ordered_values(df, value_col):
    """
    Return metric values ordered as MultiVI, GLUE, MOFA+.
    """
    method_col = find_method_column(df)
    values = []

    for method in METHODS:
        rows = df[df[method_col].astype(str).str.contains(method, case=False, regex=False)]

        if rows.empty:
            raise ValueError(
                f"Cannot find method '{method}' in column '{method_col}'. "
                f"Available values: {df[method_col].tolist()}"
            )

        values.append(float(rows.iloc[0][value_col]))

    return values


def save_figure(fig, outfile_base):
    """
    Save figure as both PNG and PDF.
    """
    png_path = os.path.join(FIG_DIR, f"{outfile_base}.png")
    pdf_path = os.path.join(FIG_DIR, f"{outfile_base}.pdf")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# 1. Paired cell-type conservation benchmark
# ============================================================
def plot_paired_celltype_conservation():
    """
    Generate paired cell-type conservation benchmark figure.

    Metrics:
    - NMI
    - Balanced silhouette
    - Balanced KNN purity
    """
    input_csv = os.path.join(RESULT_DIR, "strict_paired_benchmark_metrics.csv")
    df = pd.read_csv(input_csv)

    metric_map = {
        "NMI": "NMI",
        "Balanced silhouette": "Silhouette_balanced_celltype",
        "Balanced KNN purity": "KNN_purity_balanced_celltype",
    }

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

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

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

    ax.set_title("Paired cell-type conservation benchmark", fontweight="bold", pad=10)
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metric_map.keys()))
    ax.set_ylim(0, 1.10)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()

    save_figure(fig, "figure1D_paired_celltype_conservation_benchmark")


# ============================================================
# 2. Paired ARI
# ============================================================
def plot_paired_ari():
    """
    Generate paired ARI supplementary figure.
    """
    input_csv = os.path.join(RESULT_DIR, "paired_ari_summary.csv")
    df = pd.read_csv(input_csv)

    values = get_ordered_values(df, "ARI")

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    bars = ax.bar(
        METHODS,
        values,
        color=[COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.9,
        width=0.65,
    )

    ax.set_title("Paired ARI", fontweight="bold", pad=8)
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    save_figure(fig, "supp_paired_ari")


# ============================================================
# 3. Paired overall immune programme smoothness
# ============================================================
def plot_paired_immune_smoothness():
    """
    Generate paired overall immune programme smoothness supplementary figure.
    """
    input_csv = os.path.join(RESULT_DIR, "paired_immune_programme_smoothness_metrics.csv")
    df = pd.read_csv(input_csv)

    smooth_col = "overall_immune_programme_smoothness"
    print(f"Using immune smoothness column: {smooth_col}")

    values = get_ordered_values(df, smooth_col)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    bars = ax.bar(
        METHODS,
        values,
        color=[COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.9,
        width=0.65,
    )

    ax.set_title("Paired immune programme smoothness", fontweight="bold", pad=8)
    ax.set_ylabel("Immune smoothness")

    # Adaptive y-axis because immune smoothness values are around 0.30
    ymax = max(values) * 1.25 if max(values) > 0 else 1.0
    ax.set_ylim(0, ymax)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    save_figure(fig, "supp_paired_immune_programme_smoothness")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating paired benchmark figures...")

    plot_paired_celltype_conservation()
    plot_paired_ari()
    plot_paired_immune_smoothness()

    print("Done. All paired figures saved to figures/")
