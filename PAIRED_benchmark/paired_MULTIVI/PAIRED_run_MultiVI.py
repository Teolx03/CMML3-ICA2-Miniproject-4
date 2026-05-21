import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import scvi
import torch

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
scvi.settings.seed = SEED

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = "/public/workspace/3230300361bit/CMML_ICA2_MultiVI"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "pbmc_multiome_preprocessed.h5mu")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR = os.path.join(RESULT_DIR, "multivi_model_full")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Reading MuData...")
mdata = mu.read_h5mu(DATA_PATH)

print(mdata)
print("Modalities:", mdata.mod.keys())
print("RNA shape:", mdata.mod["rna"].shape)
print("ATAC shape:", mdata.mod["atac"].shape)
print("CUDA available:", torch.cuda.is_available())

# -----------------------------
# Setup MultiVI
# No batch_key because current object has no batch column
# -----------------------------
print("Setting up MultiVI...")

scvi.model.MULTIVI.setup_mudata(
    mdata,
    rna_layer="counts",
    atac_layer="counts",
    modalities={
        "rna_layer": "rna",
        "atac_layer": "atac",
    },
)

# -----------------------------
# Build and train model
# -----------------------------
print("Building MultiVI model...")

model = scvi.model.MULTIVI(
    mdata,
    n_latent=20,
)

print(model)

print("Training MultiVI full model...")

model.train(
    max_epochs=50,
    accelerator="auto",
    devices="auto",
)

# -----------------------------
# Save model
# -----------------------------
print("Saving model...")
model.save(MODEL_DIR, overwrite=True)

# -----------------------------
# Extract latent representation
# -----------------------------
print("Extracting latent representation...")
latent = model.get_latent_representation()

print("Latent shape:", latent.shape)

latent_df = pd.DataFrame(
    latent,
    index=mdata.obs_names,
    columns=[f"MultiVI_{i+1}" for i in range(latent.shape[1])]
)

latent_csv_path = os.path.join(RESULT_DIR, "multivi_full_latent.csv")
latent_df.to_csv(latent_csv_path)

# -----------------------------
# Create AnnData object for downstream analysis
# -----------------------------
adata_latent = sc.AnnData(X=latent)
adata_latent.obs_names = mdata.obs_names
adata_latent.obs = mdata.obs.copy()

# Add existing RNA/ATAC cluster labels if available
if "rna_leiden" in mdata.mod["rna"].obs.columns:
    adata_latent.obs["rna_leiden"] = mdata.mod["rna"].obs["rna_leiden"].astype(str).values

if "atac_leiden" in mdata.mod["atac"].obs.columns:
    adata_latent.obs["atac_leiden"] = mdata.mod["atac"].obs["atac_leiden"].astype(str).values

# -----------------------------
# UMAP and Leiden clustering
# -----------------------------
print("Running neighbors, UMAP, and Leiden clustering...")

sc.pp.neighbors(adata_latent, n_neighbors=15, use_rep="X")
sc.tl.umap(adata_latent)
sc.tl.leiden(
    adata_latent,
    resolution=0.5,
    key_added="multivi_leiden"
)

# -----------------------------
# Save latent AnnData
# -----------------------------
latent_h5ad_path = os.path.join(RESULT_DIR, "multivi_full_latent.h5ad")
adata_latent.write_h5ad(latent_h5ad_path)

# -----------------------------
# Save UMAP figures
# -----------------------------
sc.settings.figdir = FIGURE_DIR

print("Saving UMAP figures...")

sc.pl.umap(
    adata_latent,
    color=["multivi_leiden"],
    show=False,
    save="_multivi_leiden.png"
)

if "rna_leiden" in adata_latent.obs.columns:
    sc.pl.umap(
        adata_latent,
        color=["rna_leiden"],
        show=False,
        save="_rna_leiden_on_multivi.png"
    )

if "atac_leiden" in adata_latent.obs.columns:
    sc.pl.umap(
        adata_latent,
        color=["atac_leiden"],
        show=False,
        save="_atac_leiden_on_multivi.png"
    )

# -----------------------------
# Save metadata
# -----------------------------
summary_path = os.path.join(RESULT_DIR, "multivi_full_summary.txt")

with open(summary_path, "w") as f:
    f.write("MultiVI full run summary\n")
    f.write("========================\n")
    f.write(f"Input file: {DATA_PATH}\n")
    f.write(f"Cells: {mdata.n_obs}\n")
    f.write(f"RNA features: {mdata.mod['rna'].n_vars}\n")
    f.write(f"ATAC features: {mdata.mod['atac'].n_vars}\n")
    f.write(f"Latent shape: {latent.shape}\n")
    f.write(f"scvi-tools version: {scvi.__version__}\n")
    f.write(f"torch version: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    f.write(f"max_epochs: 50\n")
    f.write(f"n_latent: 20\n")
    f.write("\nOutput files:\n")
    f.write(f"- {latent_csv_path}\n")
    f.write(f"- {latent_h5ad_path}\n")
    f.write(f"- {MODEL_DIR}\n")

print("Done.")
print("Saved latent CSV:", latent_csv_path)
print("Saved latent h5ad:", latent_h5ad_path)
print("Saved model:", MODEL_DIR)
print("Saved summary:", summary_path)
