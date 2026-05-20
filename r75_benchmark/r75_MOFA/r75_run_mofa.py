import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import mudata as md
import anndata as ad
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = "/public/workspace/3230300361bit/CMML_ICA2_MultiVI"

DATA_PATH = os.path.join(PROJECT_DIR, "data/pbmc_multiome_preprocessed.h5mu")
STATUS_PATH = os.path.join(PROJECT_DIR, "results/r75_modality_status.csv")

RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_PATH = os.path.join(RESULT_DIR, "mofa_r75_model.hdf5")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# 1. Load data and r75 metadata
# ============================================================

print("Loading original fully paired MuData...")
mdata = mu.read_h5mu(DATA_PATH)

rna = mdata.mod["rna"].copy()
atac = mdata.mod["atac"].copy()

print("RNA:")
print(rna)
print("ATAC:")
print(atac)

print("Loading r75 metadata...")
status_df = pd.read_csv(STATUS_PATH)
status_df = status_df.set_index("cell")

common_cells = rna.obs_names.intersection(atac.obs_names).intersection(status_df.index)
status_df = status_df.loc[common_cells].copy()

rna = rna[common_cells].copy()
atac = atac[common_cells].copy()

print("Common cells:", len(common_cells))
print(status_df["modality_status"].value_counts())
print(status_df["final_cell_type"].value_counts())


# ============================================================
# 2. Prepare RNA view
# ============================================================

print("Preparing RNA view for MOFA+ r75...")

# Use log-normalized RNA if available
if "log_norm" in rna.layers:
    rna_X = rna.layers["log_norm"].copy()
else:
    print("RNA log_norm layer not found; creating normalized log RNA.")
    if "counts" in rna.layers:
        rna.X = rna.layers["counts"].copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    rna_X = rna.X.copy()

# Use HVGs if already available, otherwise compute
if "highly_variable" not in rna.var.columns:
    print("RNA highly_variable not found; computing HVGs.")
    if "counts" in rna.layers:
        sc.pp.highly_variable_genes(
            rna,
            n_top_genes=3000,
            flavor="seurat_v3",
            layer="counts"
        )
    else:
        sc.pp.highly_variable_genes(
            rna,
            n_top_genes=3000
        )

hvg_mask = rna.var["highly_variable"].values

# If too many HVGs or too few, enforce top 3000 if rank exists
if hvg_mask.sum() > 3000 and "highly_variable_rank" in rna.var.columns:
    hvg_mask = rna.var["highly_variable_rank"].values < 3000

rna_hvg_names = rna.var_names[hvg_mask]
rna_X = rna_X[:, hvg_mask]

# Convert to dense float matrix
if hasattr(rna_X, "toarray"):
    rna_X = rna_X.toarray()

rna_X = np.asarray(rna_X, dtype=np.float32)

print("RNA view shape before missing mask:", rna_X.shape)


# ============================================================
# 3. Prepare ATAC view
# ============================================================

print("Preparing ATAC view for MOFA+ r75...")

if "X_lsi" not in atac.obsm:
    raise ValueError("ATAC X_lsi not found. Please compute LSI first.")

atac_X = np.asarray(atac.obsm["X_lsi"][:, :50], dtype=np.float32)

atac_feature_names = [f"LSI_{i}" for i in range(atac_X.shape[1])]

print("ATAC view shape before missing mask:", atac_X.shape)


# ============================================================
# 4. Apply r75 missing modality mask using NaN
# ============================================================

print("Applying r75 missing modality mask as NaN...")

modality_status = status_df["modality_status"].astype(str).values

rna_missing = modality_status == "atac_only"
atac_missing = modality_status == "rna_only"

print("RNA missing cells:", rna_missing.sum())
print("ATAC missing cells:", atac_missing.sum())

rna_X[rna_missing, :] = np.nan
atac_X[atac_missing, :] = np.nan

print("RNA NaN count:", np.isnan(rna_X).sum())
print("ATAC NaN count:", np.isnan(atac_X).sum())


# ============================================================
# 5. Create MuData for MOFA+
# ============================================================

print("Creating MOFA+ r75 MuData...")

rna_mofa = ad.AnnData(X=rna_X)
rna_mofa.obs_names = common_cells.copy()
rna_mofa.var_names = rna_hvg_names.astype(str)

atac_mofa = ad.AnnData(X=atac_X)
atac_mofa.obs_names = common_cells.copy()
atac_mofa.var_names = atac_feature_names

for obj in [rna_mofa, atac_mofa]:
    obj.obs["modality_status"] = status_df["modality_status"].astype(str).values
    obj.obs["final_cell_type"] = status_df["final_cell_type"].astype(str).values
    obj.obs["has_rna"] = status_df["has_rna"].values
    obj.obs["has_atac"] = status_df["has_atac"].values

mdata_mofa = md.MuData({
    "rna": rna_mofa,
    "atac": atac_mofa
})

mdata_mofa.obs["modality_status"] = status_df["modality_status"].astype(str).values
mdata_mofa.obs["final_cell_type"] = status_df["final_cell_type"].astype(str).values
mdata_mofa.obs["has_rna"] = status_df["has_rna"].values
mdata_mofa.obs["has_atac"] = status_df["has_atac"].values

print(mdata_mofa)
print(mdata_mofa.mod["rna"])
print(mdata_mofa.mod["atac"])


# ============================================================
# 6. Run MOFA+
# ============================================================

print("Running MOFA+ r75...")

# Remove previous model if exists
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

mu.tl.mofa(
    mdata_mofa,
    n_factors=20,
    outfile=MODEL_PATH,
    use_obs="union"
)

print("MOFA+ model saved:", MODEL_PATH)


# ============================================================
# 7. Extract latent factors
# ============================================================

print("Extracting MOFA+ r75 latent factors...")

if "X_mofa" in mdata_mofa.obsm:
    latent = mdata_mofa.obsm["X_mofa"]
elif "X_mofa" in mdata_mofa.mod["rna"].obsm:
    latent = mdata_mofa.mod["rna"].obsm["X_mofa"]
else:
    print("Available MuData obsm keys:", list(mdata_mofa.obsm.keys()))
    print("Available RNA obsm keys:", list(mdata_mofa.mod["rna"].obsm.keys()))
    print("Available ATAC obsm keys:", list(mdata_mofa.mod["atac"].obsm.keys()))
    raise KeyError("Could not find X_mofa in MOFA+ output.")

latent = np.asarray(latent)

print("Latent shape:", latent.shape)


# ============================================================
# 8. Create AnnData, UMAP and Leiden
# ============================================================

print("Creating MOFA+ r75 AnnData...")

adata = ad.AnnData(X=latent)
adata.obs_names = common_cells.copy()

adata.obs["modality_status"] = status_df["modality_status"].astype(str).values
adata.obs["final_cell_type"] = status_df["final_cell_type"].astype(str).values
adata.obs["has_rna"] = status_df["has_rna"].values
adata.obs["has_atac"] = status_df["has_atac"].values

adata.obsm["X_mofa_r75"] = latent

print("Computing neighbours, UMAP and Leiden...")

sc.pp.neighbors(
    adata,
    use_rep="X_mofa_r75",
    n_neighbors=15
)

sc.tl.umap(adata)

sc.tl.leiden(
    adata,
    resolution=1.0,
    key_added="mofa_r75_leiden"
)


# ============================================================
# 9. Save outputs
# ============================================================

print("Saving MOFA+ r75 outputs...")

out_h5ad = os.path.join(RESULT_DIR, "mofa_r75_latent.h5ad")
out_csv = os.path.join(RESULT_DIR, "mofa_r75_latent.csv")
out_summary = os.path.join(RESULT_DIR, "mofa_r75_summary.txt")

adata.write_h5ad(out_h5ad)

pd.DataFrame(
    latent,
    index=adata.obs_names,
    columns=[f"MOFA_r75_{i}" for i in range(latent.shape[1])]
).to_csv(out_csv)

with open(out_summary, "w") as f:
    f.write("MOFA+ r75 completed successfully\n")
    f.write(f"Cells: {adata.n_obs}\n")
    f.write(f"Latent dimensions: {latent.shape[1]}\n")
    f.write(f"RNA view shape: {rna_X.shape}\n")
    f.write(f"ATAC view shape: {atac_X.shape}\n")
    f.write("RNA missing modality encoded as NaN for atac_only cells\n")
    f.write("ATAC missing modality encoded as NaN for rna_only cells\n")
    f.write("\nModality status counts:\n")
    f.write(str(adata.obs["modality_status"].value_counts()))
    f.write("\n\nFinal cell type counts:\n")
    f.write(str(adata.obs["final_cell_type"].value_counts()))


# ============================================================
# 10. Plot UMAPs
# ============================================================

print("Plotting MOFA+ r75 UMAPs...")

sc.pl.umap(
    adata,
    color="final_cell_type",
    frameon=False,
    legend_loc="right margin",
    title="MOFA+ r75: final cell type",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_mofa_r75_cell_type.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.umap(
    adata,
    color="modality_status",
    frameon=False,
    legend_loc="right margin",
    title="MOFA+ r75: modality status",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_mofa_r75_modality_status.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.umap(
    adata,
    color="mofa_r75_leiden",
    frameon=False,
    legend_loc="right margin",
    title="MOFA+ r75: Leiden clusters",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_mofa_r75_leiden.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 11. Finish
# ============================================================

print("Done.")
print("Saved files:")
print(out_h5ad)
print(out_csv)
print(out_summary)
print(os.path.join(FIG_DIR, "umap_mofa_r75_cell_type.png"))
print(os.path.join(FIG_DIR, "umap_mofa_r75_modality_status.png"))
print(os.path.join(FIG_DIR, "umap_mofa_r75_leiden.png"))
