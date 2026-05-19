import os
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import MuData
from scipy import sparse

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = "/public/workspace/3230300361bit/CMML_ICA2_MultiVI"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "pbmc_multiome_preprocessed.h5mu")
FINAL_H5AD = os.path.join(PROJECT_DIR, "results", "multivi_full_latent_annotated_final.h5ad")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print("Reading data...")
mdata = mu.read_h5mu(DATA_PATH)
adata_ref = sc.read_h5ad(FINAL_H5AD)

rna = mdata.mod["rna"].copy()
atac = mdata.mod["atac"].copy()

# -----------------------------
# Match cell order with final annotation
# -----------------------------
common = adata_ref.obs_names.intersection(rna.obs_names).intersection(atac.obs_names)
print("Common cells:", len(common))

rna = rna[common].copy()
atac = atac[common].copy()
adata_ref = adata_ref[common].copy()

# -----------------------------
# Prepare RNA view: top HVGs
# rna.X should already be normalized/log1p from preprocessing.
# -----------------------------
print("Preparing RNA view...")

if "highly_variable" in rna.var.columns:
    rna_mofa = rna[:, rna.var["highly_variable"]].copy()
else:
    print("No HVG column found; selecting HVGs now.")
    sc.pp.highly_variable_genes(rna, n_top_genes=3000, flavor="seurat")
    rna_mofa = rna[:, rna.var["highly_variable"]].copy()

# Limit to 3000 HVGs if more are present
if rna_mofa.n_vars > 3000:
    hvg_scores = rna_mofa.var["dispersions_norm"] if "dispersions_norm" in rna_mofa.var.columns else None
    if hvg_scores is not None:
        top_genes = hvg_scores.sort_values(ascending=False).head(3000).index
        rna_mofa = rna_mofa[:, top_genes].copy()
    else:
        rna_mofa = rna_mofa[:, :3000].copy()

# Scale RNA features for MOFA
sc.pp.scale(rna_mofa, max_value=10)

# -----------------------------
# Prepare ATAC view: use existing LSI dimensions
# -----------------------------
print("Preparing ATAC view...")

if "X_lsi" not in atac.obsm:
    raise ValueError("ATAC object has no X_lsi. Please run ATAC LSI preprocessing first.")

# Use first 50 LSI dimensions
X_lsi = atac.obsm["X_lsi"][:, :50]

atac_lsi = sc.AnnData(X=X_lsi)
atac_lsi.obs_names = atac.obs_names.copy()
atac_lsi.var_names = [f"LSI_{i+1}" for i in range(X_lsi.shape[1])]

# Scale LSI dimensions
sc.pp.scale(atac_lsi, max_value=10)

# -----------------------------
# Build MuData for MOFA+
# -----------------------------
print("Building MOFA input MuData...")

mofa_data = MuData({
    "rna": rna_mofa,
    "atac_lsi": atac_lsi,
})

# Add reference annotation for later plotting
mofa_data.obs["final_cell_type"] = adata_ref.obs["final_cell_type"].astype(str).values

print(mofa_data)
print("RNA MOFA shape:", mofa_data.mod["rna"].shape)
print("ATAC LSI MOFA shape:", mofa_data.mod["atac_lsi"].shape)

# -----------------------------
# Run MOFA+
# -----------------------------
print("Running MOFA+...")

# n_factors controls latent dimension.
# Start with 20 to match MultiVI n_latent=20.
mu.tl.mofa(
    mofa_data,
    n_factors=20,
    outfile=os.path.join(RESULT_DIR, "mofa_model.hdf5"),
    use_obs="union",
)

print("MOFA obsm keys:", mofa_data.obsm.keys())

# muon usually stores factors in X_mofa
if "X_mofa" not in mofa_data.obsm:
    raise ValueError(f"X_mofa not found. Available obsm keys: {mofa_data.obsm.keys()}")

# -----------------------------
# Create AnnData from MOFA factors
# -----------------------------
print("Creating MOFA latent AnnData...")

adata_mofa = sc.AnnData(X=mofa_data.obsm["X_mofa"])
adata_mofa.obs_names = mofa_data.obs_names.copy()
adata_mofa.obs["final_cell_type"] = mofa_data.obs["final_cell_type"].astype(str).values

# Neighbors, UMAP, Leiden
sc.pp.neighbors(adata_mofa, n_neighbors=15, use_rep="X")
sc.tl.umap(adata_mofa)
sc.tl.leiden(adata_mofa, resolution=0.5, key_added="mofa_leiden")

# Save object
out_h5ad = os.path.join(RESULT_DIR, "mofa_latent.h5ad")
adata_mofa.write_h5ad(out_h5ad)

# Save latent csv
latent_df = pd.DataFrame(
    adata_mofa.X,
    index=adata_mofa.obs_names,
    columns=[f"MOFA_{i+1}" for i in range(adata_mofa.X.shape[1])]
)
latent_df.to_csv(os.path.join(RESULT_DIR, "mofa_latent.csv"))

# Save plots
sc.settings.figdir = FIGURE_DIR

sc.pl.umap(
    adata_mofa,
    color=["final_cell_type"],
    show=False,
    save="_mofa_final_cell_type.png"
)

sc.pl.umap(
    adata_mofa,
    color=["mofa_leiden"],
    legend_loc="on data",
    show=False,
    save="_mofa_leiden.png"
)

print("Done.")
print("Saved:", out_h5ad)
print("Saved model:", os.path.join(RESULT_DIR, "mofa_model.hdf5"))
