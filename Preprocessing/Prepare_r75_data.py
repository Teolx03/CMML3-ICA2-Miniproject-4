import os
import numpy as np
import pandas as pd
import muon as mu
import scanpy as sc
import mudata as md
from scipy import sparse


PROJECT_DIR = "/public/workspace/3230300361bit/CMML_ICA2_MultiVI"

INPUT_H5MU = os.path.join(PROJECT_DIR, "data/pbmc_multiome_preprocessed.h5mu")
ANNOTATED_H5AD = os.path.join(PROJECT_DIR, "results/multivi_full_latent_annotated_final.h5ad")

OUT_METADATA = os.path.join(PROJECT_DIR, "results/r75_modality_status.csv")
OUT_H5MU = os.path.join(PROJECT_DIR, "data/pbmc_multiome_r75.h5mu")

np.random.seed(123)


def zero_rows_matrix(X, row_mask):
    """
    Set selected rows to zero while preserving sparse/dense format.
    row_mask: boolean array, True means set this row to zero.
    """
    X = X.copy()

    if sparse.issparse(X):
        X = X.tolil()
        X[row_mask, :] = 0
        X = X.tocsr()
        X.eliminate_zeros()
    else:
        X[row_mask, :] = 0

    return X


# ============================================================
# 1. Load fully paired data and annotation
# ============================================================

print("Loading original fully paired MuData...")
mdata = mu.read_h5mu(INPUT_H5MU)

print(mdata)
print(mdata.mod["rna"])
print(mdata.mod["atac"])

print("Loading final cell type annotation...")
annot = sc.read_h5ad(ANNOTATED_H5AD)

rna_cells = mdata.mod["rna"].obs_names
atac_cells = mdata.mod["atac"].obs_names

common_cells = rna_cells.intersection(atac_cells).intersection(annot.obs_names)
common_cells = np.array(common_cells)

n_cells = len(common_cells)
print("Number of common cells:", n_cells)


# ============================================================
# 2. Create r75 modality status
# ============================================================

n_paired = int(round(n_cells * 0.25))
n_rna_only = int(round(n_cells * 0.375))
n_atac_only = n_cells - n_paired - n_rna_only

print("Target r75 split:")
print("paired:", n_paired)
print("rna_only:", n_rna_only)
print("atac_only:", n_atac_only)

shuffled = np.random.permutation(common_cells)

paired_cells = shuffled[:n_paired]
rna_only_cells = shuffled[n_paired:n_paired + n_rna_only]
atac_only_cells = shuffled[n_paired + n_rna_only:]

status = pd.Series(index=common_cells, dtype=str)
status.loc[paired_cells] = "paired"
status.loc[rna_only_cells] = "rna_only"
status.loc[atac_only_cells] = "atac_only"

metadata = pd.DataFrame({
    "cell": common_cells,
    "modality_status": status.loc[common_cells].values,
    "has_rna": status.loc[common_cells].isin(["paired", "rna_only"]).values,
    "has_atac": status.loc[common_cells].isin(["paired", "atac_only"]).values,
    "final_cell_type": annot.obs.loc[common_cells, "final_cell_type"].astype(str).values
})

metadata.to_csv(OUT_METADATA, index=False)

print("\nSaved r75 metadata:")
print(OUT_METADATA)

print("\nModality status counts:")
print(metadata["modality_status"].value_counts())

print("\nFinal cell type counts:")
print(metadata["final_cell_type"].value_counts())


# ============================================================
# 3. Create padded r75 RNA and ATAC AnnData objects
# ============================================================

print("\nCreating padded r75 RNA and ATAC AnnData objects...")

# Keep all common cells in both modalities
rna_r75 = mdata.mod["rna"][common_cells].copy()
atac_r75 = mdata.mod["atac"][common_cells].copy()

meta_indexed = metadata.set_index("cell").loc[common_cells]

for col in ["modality_status", "has_rna", "has_atac", "final_cell_type"]:
    rna_r75.obs[col] = meta_indexed[col].values
    atac_r75.obs[col] = meta_indexed[col].values

# Define missing rows
rna_missing = meta_indexed["modality_status"].values == "atac_only"
atac_missing = meta_indexed["modality_status"].values == "rna_only"

print("RNA missing rows to zero-pad:", rna_missing.sum())
print("ATAC missing rows to zero-pad:", atac_missing.sum())

# Make sure counts layers exist
if "counts" not in rna_r75.layers:
    print("RNA counts layer not found; using X as counts.")
    rna_r75.layers["counts"] = rna_r75.X.copy()

if "counts" not in atac_r75.layers:
    print("ATAC counts layer not found; using X as counts.")
    atac_r75.layers["counts"] = atac_r75.X.copy()

# Zero-pad missing modality in X and counts layer
rna_r75.X = zero_rows_matrix(rna_r75.X, rna_missing)
rna_r75.layers["counts"] = zero_rows_matrix(rna_r75.layers["counts"], rna_missing)

atac_r75.X = zero_rows_matrix(atac_r75.X, atac_missing)
atac_r75.layers["counts"] = zero_rows_matrix(atac_r75.layers["counts"], atac_missing)

# Store missing-modality mask
rna_r75.obs["is_missing_rna"] = rna_missing
atac_r75.obs["is_missing_atac"] = atac_missing

print("RNA r75:")
print(rna_r75)

print("ATAC r75:")
print(atac_r75)

print("RNA count sum for atac_only cells should be 0:")
print(np.asarray(rna_r75.layers["counts"][rna_missing].sum()).sum())

print("ATAC count sum for rna_only cells should be 0:")
print(np.asarray(atac_r75.layers["counts"][atac_missing].sum()).sum())


# ============================================================
# 4. Save padded r75 MuData
# ============================================================

print("\nSaving padded r75 MuData...")

mdata_r75 = md.MuData({
    "rna": rna_r75,
    "atac": atac_r75
})

mdata_r75.obs["modality_status"] = meta_indexed["modality_status"].values
mdata_r75.obs["has_rna"] = meta_indexed["has_rna"].values
mdata_r75.obs["has_atac"] = meta_indexed["has_atac"].values
mdata_r75.obs["final_cell_type"] = meta_indexed["final_cell_type"].values

mdata_r75.uns["r75_design"] = {
    "description": "Artificial r75 unpaired PBMC multiome data. Missing modalities are zero-padded so that MuData remains fully paired for MultiVI.",
    "paired_fraction": 0.25,
    "rna_only_fraction": 0.375,
    "atac_only_fraction": 0.375,
    "random_seed": 123
}

mdata_r75.write_h5mu(OUT_H5MU)

print("Saved padded r75 MuData:")
print(OUT_H5MU)

print("\nFinal check:")
print(mdata_r75)
print("RNA obs:", mdata_r75.mod["rna"].n_obs)
print("ATAC obs:", mdata_r75.mod["atac"].n_obs)
print(mdata_r75.obs["modality_status"].value_counts())

print("\nDone.")

