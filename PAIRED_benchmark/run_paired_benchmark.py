import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.stats import zscore

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
RESULT_DIR = "results"
DATA_DIR = "data"

os.makedirs(RESULT_DIR, exist_ok=True)

ANNOTATED_H5AD = os.path.join(RESULT_DIR, "multivi_full_latent_annotated_final.h5ad")

LATENT_FILES = {
    "MultiVI": os.path.join(RESULT_DIR, "multivi_full_latent.csv"),
    "GLUE": os.path.join(RESULT_DIR, "glue_latent.csv"),
    "MOFA+": os.path.join(RESULT_DIR, "mofa_latent.csv"),
}

PREPROCESSED_H5MU = os.path.join(DATA_DIR, "pbmc_multiome_preprocessed.h5mu")

# ============================================================
# Fixed benchmark settings
# ============================================================
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 1.0
RANDOM_STATE = 0

METHOD_ORDER = ["MultiVI", "GLUE", "MOFA+"]

# ============================================================
# Immune programme marker genes
# ============================================================
IMMUNE_PROGRAMMES = {
    "T_cell_programme": ["CD3D", "CD3E", "CD3G", "TRAC", "IL7R"],
    "Cytotoxic_programme": ["NKG7", "GNLY", "GZMB", "PRF1", "KLRD1"],
    "B_cell_programme": ["MS4A1", "CD79A", "CD79B", "CD74"],
    "Monocyte_inflammatory_programme": ["LYZ", "LST1", "S100A8", "S100A9", "FCGR3A", "MS4A7", "CST3"],
    "Antigen_presentation_programme": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-A", "HLA-B", "HLA-C"],
}


# ============================================================
# Helper functions
# ============================================================
def read_latent_csv(path):
    """
    Read latent csv where first column is cell barcode index.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def load_final_cell_type():
    """
    Load final marker-based broad PBMC annotation from annotated MultiVI h5ad.
    """
    if not os.path.exists(ANNOTATED_H5AD):
        raise FileNotFoundError(f"Missing annotation file: {ANNOTATED_H5AD}")

    adata = sc.read_h5ad(ANNOTATED_H5AD)

    if "final_cell_type" not in adata.obs.columns:
        raise KeyError(
            f"'final_cell_type' not found in {ANNOTATED_H5AD}. "
            f"Available obs columns: {adata.obs.columns.tolist()}"
        )

    obs = adata.obs[["final_cell_type"]].copy()
    obs.index = obs.index.astype(str)

    return obs


def make_method_adata(method, latent_df, annotation_df):
    """
    Make AnnData for one method using latent embeddings and final cell type labels.
    """
    common_cells = latent_df.index.intersection(annotation_df.index)

    if len(common_cells) == 0:
        raise ValueError(f"No overlapping cells found for {method}.")

    latent_df = latent_df.loc[common_cells].copy()
    obs = annotation_df.loc[common_cells].copy()

    X = latent_df.values.astype(float)

    adata = ad.AnnData(X=X, obs=obs)
    adata.obs["final_cell_type"] = adata.obs["final_cell_type"].astype(str)
    adata.uns["method"] = method

    return adata


def run_leiden(adata, leiden_key):
    """
    Build KNN graph and run Leiden clustering.
    """
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X",
        random_state=RANDOM_STATE,
    )

    sc.tl.leiden(
        adata,
        resolution=LEIDEN_RESOLUTION,
        key_added=leiden_key,
        random_state=RANDOM_STATE,
    )

    return adata


def compute_balanced_silhouette(X, labels):
    """
    Compute ordinary and balanced silhouette.

    Balanced silhouette = average silhouette within each cell type first,
    then average across cell types equally.
    """
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        return np.nan, np.nan

    sample_sil = silhouette_samples(X, labels, metric="euclidean")
    ordinary = float(np.mean(sample_sil))

    celltype_scores = []
    for ct in sorted(np.unique(labels)):
        mask = labels == ct
        if np.sum(mask) > 0:
            celltype_scores.append(float(np.mean(sample_sil[mask])))

    balanced = float(np.mean(celltype_scores))

    return ordinary, balanced


def compute_knn_purity(X, labels, k=N_NEIGHBORS):
    """
    Compute ordinary and balanced KNN cell-type purity.
    Also return cell-type-wise purity table.
    """
    labels = np.asarray(labels)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    # remove self-neighbour
    neighbor_indices = indices[:, 1:]

    per_cell_purity = []
    for i in range(X.shape[0]):
        same = labels[neighbor_indices[i]] == labels[i]
        per_cell_purity.append(np.mean(same))

    per_cell_purity = np.asarray(per_cell_purity)
    ordinary = float(np.mean(per_cell_purity))

    rows = []
    for ct in sorted(np.unique(labels)):
        mask = labels == ct
        ct_purity = float(np.mean(per_cell_purity[mask]))
        rows.append({
            "cell_type": ct,
            "n_cells": int(np.sum(mask)),
            "KNN_purity": ct_purity,
        })

    celltype_df = pd.DataFrame(rows)
    balanced = float(celltype_df["KNN_purity"].mean())

    return ordinary, balanced, celltype_df


def calculate_method_metrics(method, adata):
    """
    Calculate ARI, NMI, silhouette and KNN purity for one method.
    """
    leiden_key = f"{method.lower().replace('+', '').replace(' ', '_')}_leiden"
    adata = run_leiden(adata, leiden_key=leiden_key)

    labels = adata.obs["final_cell_type"].astype(str).values
    clusters = adata.obs[leiden_key].astype(str).values
    X = adata.X

    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)

    sil_ordinary, sil_balanced = compute_balanced_silhouette(X, labels)

    knn_ordinary, knn_balanced, celltype_knn_df = compute_knn_purity(
        X=X,
        labels=labels,
        k=N_NEIGHBORS,
    )

    metrics = {
        "method": method,
        "setting": "paired",
        "n_cells": int(adata.n_obs),
        "n_latent_dims": int(adata.n_vars),
        "rep_key": "X",
        "leiden_key": leiden_key,
        "ARI": float(ari),
        "NMI": float(nmi),
        "Silhouette_ordinary": float(sil_ordinary),
        "Silhouette_balanced_celltype": float(sil_balanced),
        "KNN_purity_ordinary": float(knn_ordinary),
        "KNN_purity_balanced_celltype": float(knn_balanced),
    }

    celltype_knn_df.insert(0, "method", method)
    celltype_knn_df.insert(1, "setting", "paired")

    return metrics, celltype_knn_df, adata


# ============================================================
# Immune programme smoothness
# ============================================================
def load_rna_expression_for_programmes(cells):
    """
    Load RNA expression matrix from h5mu and calculate marker programme scores.

    This uses the processed RNA matrix if available.
    If the h5mu file cannot be read or markers are missing, this function raises
    a clear error.
    """
    if not os.path.exists(PREPROCESSED_H5MU):
        raise FileNotFoundError(
            f"Missing preprocessed h5mu file: {PREPROCESSED_H5MU}. "
            "Immune programme smoothness needs RNA expression input."
        )

    try:
        import muon as mu
    except ImportError as e:
        raise ImportError(
            "muon is required to read .h5mu. Please run this inside multivi_env."
        ) from e

    print(f"Reading RNA data from: {PREPROCESSED_H5MU}")
    mdata = mu.read_h5mu(PREPROCESSED_H5MU)

    if "rna" not in mdata.mod:
        raise KeyError(f"'rna' modality not found in {PREPROCESSED_H5MU}")

    rna = mdata.mod["rna"]

    common_cells = pd.Index(cells).intersection(rna.obs_names.astype(str))

    if len(common_cells) == 0:
        raise ValueError("No overlapping cells between latent embeddings and RNA modality.")

    rna = rna[common_cells].copy()

    # Use rna.X. In this project this should be the processed/log-normalized RNA matrix.
    X = rna.X
    gene_names = pd.Index(rna.var_names.astype(str))

    programme_scores = pd.DataFrame(index=common_cells)

    for programme, genes in IMMUNE_PROGRAMMES.items():
        available = [g for g in genes if g in gene_names]

        if len(available) == 0:
            print(f"Warning: no genes found for {programme}. Skipping.")
            continue

        gene_idx = gene_names.get_indexer(available)

        if sparse.issparse(X):
            score = np.asarray(X[:, gene_idx].mean(axis=1)).ravel()
        else:
            score = np.asarray(X[:, gene_idx]).mean(axis=1)

        # z-score each programme to make programmes comparable
        score = zscore(score, nan_policy="omit")
        score = np.nan_to_num(score, nan=0.0)

        programme_scores[programme] = score

        print(f"{programme}: using {len(available)} genes -> {available}")

    if programme_scores.shape[1] == 0:
        raise ValueError("No immune programme marker genes were found in RNA data.")

    return programme_scores


def compute_immune_smoothness_for_method(method, adata, programme_scores):
    """
    Immune programme smoothness:
    For each cell, calculate absolute difference in programme score between
    the cell and its K nearest neighbours in latent space.

    Lower value = nearby cells have more similar immune programme activity.
    """
    common_cells = adata.obs_names.astype(str).intersection(programme_scores.index)

    if len(common_cells) == 0:
        raise ValueError(f"No overlapping cells for immune smoothness in {method}.")

    adata_sub = adata[common_cells].copy()
    scores = programme_scores.loc[common_cells].copy()

    X_latent = adata_sub.X
    score_matrix = scores.values.astype(float)

    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric="euclidean")
    nn.fit(X_latent)
    _, indices = nn.kneighbors(X_latent)
    neighbor_indices = indices[:, 1:]

    by_programme = {}

    for j, programme in enumerate(scores.columns):
        diffs = []

        for i in range(X_latent.shape[0]):
            neighbour_vals = score_matrix[neighbor_indices[i], j]
            current_val = score_matrix[i, j]
            diffs.append(np.mean(np.abs(neighbour_vals - current_val)))

        by_programme[programme] = float(np.mean(diffs))

    overall = float(np.mean(list(by_programme.values())))

    return overall, by_programme


def run_immune_smoothness(method_adatas):
    """
    Calculate paired immune programme smoothness for all methods.
    """
    first_method = METHOD_ORDER[0]
    cells = method_adatas[first_method].obs_names.astype(str)

    programme_scores = load_rna_expression_for_programmes(cells)

    summary_rows = []
    programme_rows = []

    for method in METHOD_ORDER:
        print(f"Calculating paired immune programme smoothness for {method}...")
        overall, by_programme = compute_immune_smoothness_for_method(
            method=method,
            adata=method_adatas[method],
            programme_scores=programme_scores,
        )

        summary_rows.append({
            "setting": "paired",
            "method": method,
            "rep_key": "X",
            "overall_immune_programme_smoothness": overall,
        })

        for programme, value in by_programme.items():
            programme_rows.append({
                "setting": "paired",
                "method": method,
                "programme": programme,
                "immune_programme_smoothness": value,
            })

    summary_df = pd.DataFrame(summary_rows)
    programme_df = pd.DataFrame(programme_rows)

    return summary_df, programme_df


# ============================================================
# Main
# ============================================================
def main():
    print("==============================================")
    print("Running paired benchmark from saved embeddings")
    print("==============================================")

    annotation_df = load_final_cell_type()

    metric_rows = []
    celltype_knn_rows = []
    method_adatas = {}

    for method in METHOD_ORDER:
        print(f"\nProcessing {method}...")

        latent_path = LATENT_FILES[method]

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Missing latent file for {method}: {latent_path}")

        latent_df = read_latent_csv(latent_path)
        adata = make_method_adata(method, latent_df, annotation_df)

        metrics, celltype_knn_df, adata = calculate_method_metrics(method, adata)

        metric_rows.append(metrics)
        celltype_knn_rows.append(celltype_knn_df)
        method_adatas[method] = adata

        print(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    celltype_knn_df = pd.concat(celltype_knn_rows, axis=0, ignore_index=True)

    # Keep method order fixed
    metrics_df["method"] = pd.Categorical(metrics_df["method"], categories=METHOD_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values("method").reset_index(drop=True)

    # Save main paired benchmark metrics
    metrics_out = os.path.join(RESULT_DIR, "strict_paired_benchmark_metrics.csv")
    celltype_knn_out = os.path.join(RESULT_DIR, "strict_paired_celltypewise_knn.csv")
    ari_out = os.path.join(RESULT_DIR, "paired_ari_summary.csv")

    metrics_df.to_csv(metrics_out, index=False)
    celltype_knn_df.to_csv(celltype_knn_out, index=False)

    ari_df = metrics_df[["setting", "method", "leiden_key", "ARI"]].copy()
    ari_df = ari_df.rename(columns={"leiden_key": "leiden_column"})
    ari_df.to_csv(ari_out, index=False)

    print(f"\nSaved: {metrics_out}")
    print(f"Saved: {celltype_knn_out}")
    print(f"Saved: {ari_out}")

    # Save immune programme smoothness
    try:
        immune_summary_df, immune_by_programme_df = run_immune_smoothness(method_adatas)

        immune_summary_out = os.path.join(
            RESULT_DIR,
            "paired_immune_programme_smoothness_metrics.csv",
        )
        immune_by_programme_out = os.path.join(
            RESULT_DIR,
            "paired_immune_programme_smoothness_by_programme.csv",
        )

        immune_summary_df.to_csv(immune_summary_out, index=False)
        immune_by_programme_df.to_csv(immune_by_programme_out, index=False)

        print(f"Saved: {immune_summary_out}")
        print(f"Saved: {immune_by_programme_out}")

    except Exception as e:
        print("\nWarning: immune programme smoothness was not calculated.")
        print(f"Reason: {e}")

    print("\nDone. Paired benchmark outputs regenerated.")


if __name__ == "__main__":
    main()
