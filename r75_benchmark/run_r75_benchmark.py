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

R75_STATUS_CSV = os.path.join(RESULT_DIR, "r75_modality_status.csv")
PREPROCESSED_H5MU = os.path.join(DATA_DIR, "pbmc_multiome_preprocessed.h5mu")

LATENT_FILES = {
    "MultiVI": os.path.join(RESULT_DIR, "multivi_r75_latent.csv"),
    "GLUE": os.path.join(RESULT_DIR, "glue_r75_latent.csv"),
    "MOFA+": os.path.join(RESULT_DIR, "mofa_r75_latent.csv"),
}

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
# Basic loading functions
# ============================================================
def read_latent_csv(path):
    """
    Read latent csv where first column is cell barcode index.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def load_r75_metadata():
    """
    Load r75 modality status and final cell type.
    Required columns:
    - cell
    - modality_status
    - final_cell_type
    """
    if not os.path.exists(R75_STATUS_CSV):
        raise FileNotFoundError(f"Missing r75 metadata file: {R75_STATUS_CSV}")

    meta = pd.read_csv(R75_STATUS_CSV)

    required_cols = ["cell", "modality_status", "final_cell_type"]
    missing = [c for c in required_cols if c not in meta.columns]

    if missing:
        raise KeyError(
            f"Missing columns in {R75_STATUS_CSV}: {missing}. "
            f"Available columns: {meta.columns.tolist()}"
        )

    meta["cell"] = meta["cell"].astype(str)
    meta = meta.set_index("cell")

    meta["modality_status"] = meta["modality_status"].astype(str)
    meta["final_cell_type"] = meta["final_cell_type"].astype(str)

    return meta


def make_method_adata(method, latent_df, meta_df):
    """
    Make AnnData for one r75 method using latent embeddings,
    final cell type and modality status labels.
    """
    common_cells = latent_df.index.intersection(meta_df.index)

    if len(common_cells) == 0:
        raise ValueError(f"No overlapping cells found for {method}.")

    latent_df = latent_df.loc[common_cells].copy()
    obs = meta_df.loc[common_cells, ["modality_status", "final_cell_type"]].copy()

    X = latent_df.values.astype(float)

    adata = ad.AnnData(X=X, obs=obs)
    adata.obs["final_cell_type"] = adata.obs["final_cell_type"].astype(str)
    adata.obs["modality_status"] = adata.obs["modality_status"].astype(str)
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


# ============================================================
# Cell-type conservation metrics
# ============================================================
def compute_balanced_silhouette(X, labels):
    """
    Ordinary silhouette = average across all cells.
    Balanced silhouette = average within each cell type first,
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
    Ordinary KNN purity = average across all cells.
    Balanced KNN purity = average within each cell type first,
    then average across cell types equally.
    """
    labels = np.asarray(labels)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

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


# ============================================================
# r75 cross-modal and modality mixing metrics
# ============================================================
def compute_cross_modal_knn_metrics(X, celltype_labels, modality_labels, k=N_NEIGHBORS):
    """
    Calculate:
    - cross-modal KNN purity:
      among neighbours from different modality-status groups, fraction with same cell type.
    - cross-modal cell-type agreement:
      same definition here, kept as a separately named output for interpretability.
    - KNN modality entropy:
      normalized entropy of modality-status composition in each local neighbourhood.
    - same-modality enrichment:
      observed same-modality neighbour fraction / expected same-modality fraction.
    """
    celltype_labels = np.asarray(celltype_labels)
    modality_labels = np.asarray(modality_labels)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    neighbor_indices = indices[:, 1:]

    modality_categories = sorted(np.unique(modality_labels))
    n_modality = len(modality_categories)

    cross_modal_purity_per_cell = []
    entropy_per_cell = []
    same_modality_flags = []

    for i in range(X.shape[0]):
        nbr_idx = neighbor_indices[i]

        nbr_celltypes = celltype_labels[nbr_idx]
        nbr_modalities = modality_labels[nbr_idx]

        current_celltype = celltype_labels[i]
        current_modality = modality_labels[i]

        # Cross-modal neighbours only
        cross_mask = nbr_modalities != current_modality

        if np.sum(cross_mask) > 0:
            cross_purity = np.mean(nbr_celltypes[cross_mask] == current_celltype)
            cross_modal_purity_per_cell.append(float(cross_purity))

        # Modality entropy
        counts = np.array([np.sum(nbr_modalities == m) for m in modality_categories], dtype=float)
        probs = counts / counts.sum()

        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))

        if n_modality > 1:
            entropy = entropy / np.log(n_modality)

        entropy_per_cell.append(float(entropy))

        # Same modality neighbour fraction
        same_modality_flags.append(np.mean(nbr_modalities == current_modality))

    cross_modal_knn_purity = float(np.mean(cross_modal_purity_per_cell))
    cross_modal_celltype_agreement = cross_modal_knn_purity

    knn_modality_entropy = float(np.mean(entropy_per_cell))

    observed_same_modality_fraction = float(np.mean(same_modality_flags))

    modality_freq = pd.Series(modality_labels).value_counts(normalize=True)
    expected_same_modality_fraction = float(np.sum(modality_freq.values ** 2))

    same_modality_enrichment = (
        observed_same_modality_fraction / expected_same_modality_fraction
        if expected_same_modality_fraction > 0 else np.nan
    )

    return {
        "Cross_modal_KNN_purity": cross_modal_knn_purity,
        "Cross_modal_celltype_agreement": cross_modal_celltype_agreement,
        "KNN_modality_entropy": knn_modality_entropy,
        "observed_same_modality_fraction": observed_same_modality_fraction,
        "expected_same_modality_fraction": expected_same_modality_fraction,
        "same_modality_enrichment": float(same_modality_enrichment),
    }


def compute_modality_status_silhouette(X, modality_labels):
    """
    Modality-status silhouette.
    Values close to 0 mean weak global separation by modality status.
    """
    modality_labels = np.asarray(modality_labels)

    if len(np.unique(modality_labels)) < 2:
        return np.nan

    sample_sil = silhouette_samples(X, modality_labels, metric="euclidean")
    return float(np.mean(sample_sil))


def compute_direct_rna_to_atac_agreement(method, adata, k=N_NEIGHBORS):
    """
    Direct RNA-to-ATAC agreement:
    For each RNA-only cell, find K nearest ATAC-only cells.
    Agreement = fraction of ATAC-only neighbours with the same final cell type.

    Balanced score = calculate agreement within each RNA-only cell type,
    then average across cell types equally.
    """
    X = adata.X
    obs = adata.obs.copy()

    rna_mask = obs["modality_status"].values == "rna_only"
    atac_mask = obs["modality_status"].values == "atac_only"

    if np.sum(rna_mask) == 0 or np.sum(atac_mask) == 0:
        raise ValueError(f"{method}: RNA-only or ATAC-only cells missing.")

    X_rna = X[rna_mask]
    X_atac = X[atac_mask]

    rna_labels = obs.loc[rna_mask, "final_cell_type"].astype(str).values
    atac_labels = obs.loc[atac_mask, "final_cell_type"].astype(str).values

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_atac)
    _, indices = nn.kneighbors(X_rna)

    per_cell_agreement = []

    for i in range(X_rna.shape[0]):
        neighbour_labels = atac_labels[indices[i]]
        agreement = np.mean(neighbour_labels == rna_labels[i])
        per_cell_agreement.append(float(agreement))

    per_cell_agreement = np.asarray(per_cell_agreement)

    raw_agreement = float(np.mean(per_cell_agreement))

    rows = []
    for ct in sorted(np.unique(rna_labels)):
        mask = rna_labels == ct
        ct_agreement = float(np.mean(per_cell_agreement[mask]))

        rows.append({
            "method": method,
            "setting": "r75",
            "cell_type": ct,
            "n_rna_only_cells": int(np.sum(mask)),
            "direct_rna_to_atac_agreement": ct_agreement,
        })

    celltype_df = pd.DataFrame(rows)
    balanced_agreement = float(celltype_df["direct_rna_to_atac_agreement"].mean())

    summary = {
        "setting": "r75",
        "method": method,
        "k": int(k),
        "n_rna_only": int(np.sum(rna_mask)),
        "n_atac_only": int(np.sum(atac_mask)),
        "raw_direct_rna_to_atac_agreement": raw_agreement,
        "balanced_direct_rna_to_atac_agreement": balanced_agreement,
    }

    return summary, celltype_df


# ============================================================
# Main r75 metrics
# ============================================================
def calculate_method_metrics(method, adata):
    """
    Calculate all r75 benchmark metrics for one method.
    """
    leiden_key = f"{method.lower().replace('+', '').replace(' ', '_')}_r75_leiden"
    adata = run_leiden(adata, leiden_key=leiden_key)

    labels = adata.obs["final_cell_type"].astype(str).values
    modality = adata.obs["modality_status"].astype(str).values
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

    cross_metrics = compute_cross_modal_knn_metrics(
        X=X,
        celltype_labels=labels,
        modality_labels=modality,
        k=N_NEIGHBORS,
    )

    modality_silhouette = compute_modality_status_silhouette(X, modality)

    n_paired = int(np.sum(modality == "paired"))
    n_rna_only = int(np.sum(modality == "rna_only"))
    n_atac_only = int(np.sum(modality == "atac_only"))

    metrics = {
        "method": method,
        "setting": "r75",
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
        "n_paired": n_paired,
        "n_rna_only": n_rna_only,
        "n_atac_only": n_atac_only,
        "Modality_status_silhouette": float(modality_silhouette),
    }

    metrics.update(cross_metrics)

    celltype_knn_df.insert(0, "method", method)
    celltype_knn_df.insert(1, "setting", "r75")

    return metrics, celltype_knn_df, adata


# ============================================================
# Immune programme smoothness
# ============================================================
def load_rna_expression_for_programmes(cells):
    """
    Load RNA expression matrix from h5mu and calculate immune programme scores.
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
        raise ValueError("No overlapping cells between r75 latent embeddings and RNA modality.")

    rna = rna[common_cells].copy()

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
    mean absolute difference in immune programme score between each cell
    and its K nearest neighbours in latent space.

    Lower value = local neighbours have more similar immune programme activity.
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
    Calculate r75 immune programme smoothness for all methods.
    """
    first_method = METHOD_ORDER[0]
    cells = method_adatas[first_method].obs_names.astype(str)

    programme_scores = load_rna_expression_for_programmes(cells)

    summary_rows = []
    programme_rows = []

    for method in METHOD_ORDER:
        print(f"Calculating r75 immune programme smoothness for {method}...")

        overall, by_programme = compute_immune_smoothness_for_method(
            method=method,
            adata=method_adatas[method],
            programme_scores=programme_scores,
        )

        summary_rows.append({
            "setting": "r75",
            "method": method,
            "rep_key": "X",
            "overall_immune_programme_smoothness": overall,
        })

        for programme, value in by_programme.items():
            programme_rows.append({
                "setting": "r75",
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
    print("==========================================")
    print("Running r75 benchmark from saved embeddings")
    print("==========================================")

    meta_df = load_r75_metadata()

    metric_rows = []
    celltype_knn_rows = []
    direct_summary_rows = []
    direct_celltype_rows = []
    method_adatas = {}

    for method in METHOD_ORDER:
        print(f"\nProcessing {method}...")

        latent_path = LATENT_FILES[method]

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Missing latent file for {method}: {latent_path}")

        latent_df = read_latent_csv(latent_path)
        adata = make_method_adata(method, latent_df, meta_df)

        metrics, celltype_knn_df, adata = calculate_method_metrics(method, adata)

        metric_rows.append(metrics)
        celltype_knn_rows.append(celltype_knn_df)
        method_adatas[method] = adata

        direct_summary, direct_celltype_df = compute_direct_rna_to_atac_agreement(
            method=method,
            adata=adata,
            k=N_NEIGHBORS,
        )

        direct_summary_rows.append(direct_summary)
        direct_celltype_rows.append(direct_celltype_df)

        print(metrics)
        print(direct_summary)

    metrics_df = pd.DataFrame(metric_rows)
    celltype_knn_df = pd.concat(celltype_knn_rows, axis=0, ignore_index=True)
    direct_summary_df = pd.DataFrame(direct_summary_rows)
    direct_celltype_df = pd.concat(direct_celltype_rows, axis=0, ignore_index=True)

    metrics_df["method"] = pd.Categorical(metrics_df["method"], categories=METHOD_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values("method").reset_index(drop=True)

    direct_summary_df["method"] = pd.Categorical(direct_summary_df["method"], categories=METHOD_ORDER, ordered=True)
    direct_summary_df = direct_summary_df.sort_values("method").reset_index(drop=True)

    # Main r75 outputs
    metrics_out = os.path.join(RESULT_DIR, "strict_r75_benchmark_metrics.csv")
    celltype_knn_out = os.path.join(RESULT_DIR, "strict_r75_celltypewise_knn.csv")
    ari_out = os.path.join(RESULT_DIR, "r75_ari_summary.csv")
    direct_summary_out = os.path.join(RESULT_DIR, "r75_direct_rna_to_atac_agreement_K15_summary.csv")
    direct_celltype_out = os.path.join(RESULT_DIR, "r75_direct_rna_to_atac_agreement_K15_celltypewise.csv")

    metrics_df.to_csv(metrics_out, index=False)
    celltype_knn_df.to_csv(celltype_knn_out, index=False)

    ari_df = metrics_df[["setting", "method", "leiden_key", "ARI"]].copy()
    ari_df = ari_df.rename(columns={"leiden_key": "leiden_column"})
    ari_df.to_csv(ari_out, index=False)

    direct_summary_df.to_csv(direct_summary_out, index=False)
    direct_celltype_df.to_csv(direct_celltype_out, index=False)

    print(f"\nSaved: {metrics_out}")
    print(f"Saved: {celltype_knn_out}")
    print(f"Saved: {ari_out}")
    print(f"Saved: {direct_summary_out}")
    print(f"Saved: {direct_celltype_out}")

    # Immune programme smoothness
    try:
        immune_summary_df, immune_by_programme_df = run_immune_smoothness(method_adatas)

        immune_summary_out = os.path.join(
            RESULT_DIR,
            "r75_immune_programme_smoothness_metrics.csv",
        )
        immune_by_programme_out = os.path.join(
            RESULT_DIR,
            "r75_immune_programme_smoothness_by_programme.csv",
        )

        immune_summary_df.to_csv(immune_summary_out, index=False)
        immune_by_programme_df.to_csv(immune_by_programme_out, index=False)

        print(f"Saved: {immune_summary_out}")
        print(f"Saved: {immune_by_programme_out}")

    except Exception as e:
        print("\nWarning: r75 immune programme smoothness was not calculated.")
        print(f"Reason: {e}")

    print("\nDone. r75 benchmark outputs regenerated.")


if __name__ == "__main__":
    main()

