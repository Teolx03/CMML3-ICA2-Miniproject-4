import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import anndata as ad
import matplotlib.pyplot as plt

import scvi
from scvi.model import MULTIVI


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = "/public/workspace/3230300361bit/CMML_ICA2_MultiVI"

DATA_PATH = os.path.join(PROJECT_DIR, "data/pbmc_multiome_r75.h5mu")
STATUS_PATH = os.path.join(PROJECT_DIR, "results/r75_modality_status.csv")

RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR = os.path.join(PROJECT_DIR, "results/multivi_model_r75")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. Load r75 MuData
# ============================================================

print("Loading r75 MuData...")
mdata = mu.read_h5mu(DATA_PATH)

print(mdata)
print("RNA:")
print(mdata.mod["rna"])
print("ATAC:")
print(mdata.mod["atac"])

# Important for MuData consistency
mdata.update()

print("Global MuData obs:", mdata.n_obs)
print("RNA cells:", mdata.mod["rna"].n_obs)
print("ATAC cells:", mdata.mod["atac"].n_obs)


# ============================================================
# 2. Load r75 metadata
# ============================================================

print("Loading r75 metadata...")
status_df = pd.read_csv(STATUS_PATH)
status_df = status_df.set_index("cell")

common_cells = mdata.obs_names.intersection(status_df.index)
mdata = mdata[common_cells].copy()
mdata.update()

status_df = status_df.loc[mdata.obs_names]

mdata.obs["modality_status"] = status_df["modality_status"].astype(str).values
mdata.obs["final_cell_type"] = status_df["final_cell_type"].astype(str).values
mdata.obs["has_rna"] = status_df["has_rna"].values
mdata.obs["has_atac"] = status_df["has_atac"].values

print("After aligning metadata:")
print(mdata)
print(mdata.obs["modality_status"].value_counts())
print(mdata.obs["final_cell_type"].value_counts())


# ============================================================
# 3. Make sure count layers exist
# ============================================================

print("Checking count layers...")

if "counts" not in mdata.mod["rna"].layers:
    print("Warning: RNA counts layer missing. Using RNA X as counts.")
    mdata.mod["rna"].layers["counts"] = mdata.mod["rna"].X.copy()

if "counts" not in mdata.mod["atac"].layers:
    print("Warning: ATAC counts layer missing. Using ATAC X as counts.")
    mdata.mod["atac"].layers["counts"] = mdata.mod["atac"].X.copy()


# ============================================================
# 4. Setup and train MultiVI
# ============================================================

print("scvi-tools version:", scvi.__version__)
print("Setting up MultiVI...")

MULTIVI.setup_mudata(
    mdata,
    rna_layer="counts",
    atac_layer="counts",
    modalities={
        "rna_layer": "rna",
        "atac_layer": "atac"
    }
)

print("Creating MultiVI model...")

model = MULTIVI(
    mdata,
    n_latent=20
)

print(model)

print("Training MultiVI r75 model...")

model.train(
    max_epochs=50
)

print("Saving MultiVI r75 model...")
model.save(MODEL_DIR, overwrite=True)


# ============================================================
# 5. Get latent representation
# ============================================================

print("Extracting latent representation...")

latent = model.get_latent_representation()

print("Latent shape:", latent.shape)

adata = ad.AnnData(X=latent)
adata.obs_names = mdata.obs_names.copy()

adata.obs["modality_status"] = mdata.obs["modality_status"].astype(str).values
adata.obs["final_cell_type"] = mdata.obs["final_cell_type"].astype(str).values
adata.obs["has_rna"] = mdata.obs["has_rna"].values
adata.obs["has_atac"] = mdata.obs["has_atac"].values

adata.obsm["X_multivi_r75"] = latent


# ============================================================
# 6. UMAP and Leiden
# ============================================================

print("Computing neighbours, UMAP and Leiden...")

sc.pp.neighbors(
    adata,
    use_rep="X_multivi_r75",
    n_neighbors=15
)

sc.tl.umap(adata)

sc.tl.leiden(
    adata,
    resolution=1.0,
    key_added="multivi_r75_leiden"
)


# ============================================================
# 7. Save outputs
# ============================================================

print("Saving outputs...")

out_h5ad = os.path.join(RESULT_DIR, "multivi_r75_latent.h5ad")
out_csv = os.path.join(RESULT_DIR, "multivi_r75_latent.csv")
out_summary = os.path.join(RESULT_DIR, "multivi_r75_summary.txt")

adata.write_h5ad(out_h5ad)

pd.DataFrame(
    latent,
    index=adata.obs_names,
    columns=[f"MultiVI_r75_{i}" for i in range(latent.shape[1])]
).to_csv(out_csv)

with open(out_summary, "w") as f:
    f.write("MultiVI r75 completed successfully\n")
    f.write(f"Cells: {adata.n_obs}\n")
    f.write(f"Latent dimensions: {latent.shape[1]}\n")
    f.write("\nModality status counts:\n")
    f.write(str(adata.obs["modality_status"].value_counts()))
    f.write("\n\nFinal cell type counts:\n")
    f.write(str(adata.obs["final_cell_type"].value_counts()))
    f.write("\n\nModel setting:\n")
    f.write("n_latent=20\n")
    f.write("max_epochs=50\n")


# ============================================================
# 8. Plot figures
# ============================================================

print("Plotting UMAPs...")

sc.pl.umap(
    adata,
    color="final_cell_type",
    frameon=False,
    legend_loc="right margin",
    title="MultiVI r75: final cell type",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_multivi_r75_cell_type.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.umap(
    adata,
    color="modality_status",
    frameon=False,
    legend_loc="right margin",
    title="MultiVI r75: modality status",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_multivi_r75_modality_status.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.umap(
    adata,
    color="multivi_r75_leiden",
    frameon=False,
    legend_loc="right margin",
    title="MultiVI r75: Leiden clusters",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_multivi_r75_leiden.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 9. Finish
# ============================================================

print("Done.")
print("Saved files:")
print(out_h5ad)
print(out_csv)
print(out_summary)
print(os.path.join(FIG_DIR, "umap_multivi_r75_cell_type.png"))
print(os.path.join(FIG_DIR, "umap_multivi_r75_modality_status.png"))
print(os.path.join(FIG_DIR, "umap_multivi_r75_leiden.png"))
