import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import anndata as ad
import scglue
import scglue.data
import scglue.genomics
import scglue.models
import scglue.graph
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = "/public/workspace/3230300361bit/CMML_ICA2_MultiVI"

# Use original preprocessed data, not zero-padded r75 data
DATA_PATH = os.path.join(PROJECT_DIR, "data/pbmc_multiome_preprocessed.h5mu")
STATUS_PATH = os.path.join(PROJECT_DIR, "results/r75_modality_status.csv")
GTF_PATH = os.path.join(PROJECT_DIR, "reference/gencode.v44.annotation.gtf.gz")

RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR = os.path.join(PROJECT_DIR, "results/glue_model_r75")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. Load data and r75 metadata
# ============================================================

print("Loading original fully paired MuData...")
mdata = mu.read_h5mu(DATA_PATH)

rna_full = mdata.mod["rna"].copy()
atac_full = mdata.mod["atac"].copy()

print("RNA full:")
print(rna_full)
print("ATAC full:")
print(atac_full)

print("Loading r75 modality status...")
status_df = pd.read_csv(STATUS_PATH)
status_df = status_df.set_index("cell")

common_cells = rna_full.obs_names.intersection(atac_full.obs_names).intersection(status_df.index)
status_df = status_df.loc[common_cells].copy()

print("r75 status counts:")
print(status_df["modality_status"].value_counts())

print("final cell type counts:")
print(status_df["final_cell_type"].value_counts())


# ============================================================
# 2. Create GLUE r75 modality-specific inputs
# ============================================================

rna_cells = status_df.index[status_df["has_rna"].astype(bool)]
atac_cells = status_df.index[status_df["has_atac"].astype(bool)]

print("RNA cells for GLUE r75:", len(rna_cells))
print("ATAC cells for GLUE r75:", len(atac_cells))

rna = rna_full[rna_cells].copy()
atac = atac_full[atac_cells].copy()

# Add metadata
for obj, cells in [(rna, rna_cells), (atac, atac_cells)]:
    meta = status_df.loc[cells]
    obj.obs["modality_status"] = meta["modality_status"].astype(str).values
    obj.obs["final_cell_type"] = meta["final_cell_type"].astype(str).values
    obj.obs["has_rna"] = meta["has_rna"].values
    obj.obs["has_atac"] = meta["has_atac"].values

print("RNA r75 input:")
print(rna)
print("ATAC r75 input:")
print(atac)


# ============================================================
# 3. Use raw count layers
# ============================================================

print("Preparing count matrices...")

if "counts" in rna.layers:
    rna.X = rna.layers["counts"].copy()
else:
    print("Warning: RNA counts layer not found. Using current RNA X as counts.")

if "counts" in atac.layers:
    atac.X = atac.layers["counts"].copy()
else:
    print("Warning: ATAC counts layer not found. Using current ATAC X as counts.")


# ============================================================
# 4. RNA preprocessing
# ============================================================

print("Preprocessing RNA...")

rna.layers["counts"] = rna.X.copy()

sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
rna.layers["log_norm"] = rna.X.copy()

sc.pp.highly_variable_genes(
    rna,
    n_top_genes=3000,
    flavor="seurat_v3",
    layer="counts"
)

sc.pp.scale(rna, max_value=10)

sc.tl.pca(
    rna,
    n_comps=100,
    use_highly_variable=True,
    svd_solver="arpack"
)


# ============================================================
# 5. ATAC preprocessing
# ============================================================

print("Preprocessing ATAC...")

atac.layers["counts"] = atac.X.copy()

if "X_lsi" not in atac.obsm:
    print("X_lsi not found. Computing LSI...")
    scglue.data.lsi(atac, n_components=100, n_iter=15)
else:
    print("Using existing ATAC X_lsi.")

print("Selecting top ATAC peaks...")

peak_counts = np.asarray(atac.layers["counts"].sum(axis=0)).ravel()
top_n_peaks = 50000

if len(peak_counts) < top_n_peaks:
    top_n_peaks = len(peak_counts)

top_peak_idx = np.argsort(peak_counts)[::-1][:top_n_peaks]

atac.var["highly_variable"] = False
atac.var.iloc[top_peak_idx, atac.var.columns.get_loc("highly_variable")] = True

print("Selected ATAC peaks:", top_n_peaks)


# ============================================================
# 6. Prepare RNA gene coordinates
# ============================================================

print("Preparing RNA gene annotation...")

rna.var["gene_symbols"] = rna.var_names

scglue.data.get_gene_annotation(
    rna,
    gtf=GTF_PATH,
    gtf_by="gene_name"
)

rna = rna[:, rna.var["chrom"].notna()].copy()
rna = rna[:, rna.var["highly_variable"]].copy()

print("RNA after gene annotation and HVG filtering:")
print(rna)


# ============================================================
# 7. Prepare ATAC peak coordinates
# ============================================================

print("Preparing ATAC peak coordinates...")

peak_df = atac.var_names.to_series(index=atac.var_names).str.extract(
    r"^(?P<chrom>[^:]+):(?P<chromStart>\d+)-(?P<chromEnd>\d+)$"
)

if peak_df.isna().any().any():
    print("Some ATAC peak names could not be parsed. Examples:")
    bad_peaks = peak_df.isna().any(axis=1)
    print(atac.var_names[bad_peaks][:10])
    raise ValueError("ATAC peak name format is not chr:start-end")

atac.var["chrom"] = peak_df["chrom"].values
atac.var["chromStart"] = peak_df["chromStart"].astype(int).values
atac.var["chromEnd"] = peak_df["chromEnd"].astype(int).values

atac = atac[:, atac.var["highly_variable"]].copy()

print("ATAC after peak filtering:")
print(atac)


# ============================================================
# 8. Configure GLUE datasets
# ============================================================

print("Configuring datasets for GLUE r75...")

scglue.models.configure_dataset(
    rna,
    "NB",
    use_highly_variable=True,
    use_layer="counts",
    use_rep="X_pca"
)

scglue.models.configure_dataset(
    atac,
    "NB",
    use_highly_variable=True,
    use_layer="counts",
    use_rep="X_lsi"
)


# ============================================================
# 9. Build guidance graph
# ============================================================

print("Building RNA-anchored guidance graph...")

guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)

print(guidance)
print("Number of guidance graph edges:", guidance.number_of_edges())

scglue.graph.check_graph(guidance, [rna, atac])


# ============================================================
# 10. Train GLUE r75
# ============================================================

print("Training GLUE r75 model...")

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac},
    guidance,
    fit_kws={
        "directory": MODEL_DIR,
        "max_epochs": 100,
        "patience": 15,
        "reduce_lr_patience": 5
    }
)

glue.save(os.path.join(MODEL_DIR, "final.dill"))


# ============================================================
# 11. Encode latent embeddings
# ============================================================

print("Encoding GLUE r75 latent embeddings...")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

rna_latent = pd.DataFrame(
    rna.obsm["X_glue"],
    index=rna.obs_names
)

atac_latent = pd.DataFrame(
    atac.obsm["X_glue"],
    index=atac.obs_names
)

latent_dim = rna_latent.shape[1]

all_cells = status_df.index.copy()
glue_latent = pd.DataFrame(
    np.zeros((len(all_cells), latent_dim)),
    index=all_cells,
    columns=[f"GLUE_r75_{i}" for i in range(latent_dim)]
)

# For paired cells, average RNA-side and ATAC-side embeddings
paired_cells = status_df.index[status_df["modality_status"] == "paired"]
rna_only_cells = status_df.index[status_df["modality_status"] == "rna_only"]
atac_only_cells = status_df.index[status_df["modality_status"] == "atac_only"]

glue_latent.loc[paired_cells] = (
    rna_latent.loc[paired_cells].values + atac_latent.loc[paired_cells].values
) / 2

glue_latent.loc[rna_only_cells] = rna_latent.loc[rna_only_cells].values
glue_latent.loc[atac_only_cells] = atac_latent.loc[atac_only_cells].values

print("Combined GLUE r75 latent shape:", glue_latent.shape)


# ============================================================
# 12. Create AnnData, UMAP and Leiden
# ============================================================

print("Creating combined GLUE r75 AnnData...")

adata = ad.AnnData(X=glue_latent.values)
adata.obs_names = glue_latent.index.copy()

adata.obs["modality_status"] = status_df.loc[adata.obs_names, "modality_status"].astype(str).values
adata.obs["final_cell_type"] = status_df.loc[adata.obs_names, "final_cell_type"].astype(str).values
adata.obs["has_rna"] = status_df.loc[adata.obs_names, "has_rna"].values
adata.obs["has_atac"] = status_df.loc[adata.obs_names, "has_atac"].values

adata.obsm["X_glue_r75"] = glue_latent.values

print("Computing neighbours, UMAP and Leiden...")

sc.pp.neighbors(
    adata,
    use_rep="X_glue_r75",
    n_neighbors=15
)

sc.tl.umap(adata)

sc.tl.leiden(
    adata,
    resolution=1.0,
    key_added="glue_r75_leiden"
)


# ============================================================
# 13. Save outputs
# ============================================================

print("Saving GLUE r75 outputs...")

out_h5ad = os.path.join(RESULT_DIR, "glue_r75_latent.h5ad")
out_csv = os.path.join(RESULT_DIR, "glue_r75_latent.csv")
out_summary = os.path.join(RESULT_DIR, "glue_r75_summary.txt")

adata.write_h5ad(out_h5ad)
glue_latent.to_csv(out_csv)

with open(out_summary, "w") as f:
    f.write("GLUE r75 completed successfully\n")
    f.write(f"Cells: {adata.n_obs}\n")
    f.write(f"Latent dimensions: {latent_dim}\n")
    f.write(f"RNA input cells: {rna.n_obs}\n")
    f.write(f"ATAC input cells: {atac.n_obs}\n")
    f.write(f"RNA input genes: {rna.n_vars}\n")
    f.write(f"ATAC input peaks: {atac.n_vars}\n")
    f.write(f"Guidance graph edges: {guidance.number_of_edges()}\n")
    f.write("\nModality status counts:\n")
    f.write(str(adata.obs["modality_status"].value_counts()))
    f.write("\n\nFinal cell type counts:\n")
    f.write(str(adata.obs["final_cell_type"].value_counts()))
    f.write("\n\nModel setting:\n")
    f.write("max_epochs=100 with early stopping\n")


# ============================================================
# 14. Plot UMAPs
# ============================================================

print("Plotting GLUE r75 UMAPs...")

sc.pl.umap(
    adata,
    color="final_cell_type",
    frameon=False,
    legend_loc="right margin",
    title="GLUE r75: final cell type",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_glue_r75_cell_type.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.umap(
    adata,
    color="modality_status",
    frameon=False,
    legend_loc="right margin",
    title="GLUE r75: modality status",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_glue_r75_modality_status.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.umap(
    adata,
    color="glue_r75_leiden",
    frameon=False,
    legend_loc="right margin",
    title="GLUE r75: Leiden clusters",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_glue_r75_leiden.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 15. Finish
# ============================================================

print("Done.")
print("Saved files:")
print(out_h5ad)
print(out_csv)
print(out_summary)
print(os.path.join(FIG_DIR, "umap_glue_r75_cell_type.png"))
print(os.path.join(FIG_DIR, "umap_glue_r75_modality_status.png"))
print(os.path.join(FIG_DIR, "umap_glue_r75_leiden.png"))
