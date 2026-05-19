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
# Project paths
# ============================================================

PROJECT_DIR = "/public/workspace/3230300361bit/CMML_ICA2_MultiVI"

DATA_PATH = os.path.join(PROJECT_DIR, "data/pbmc_multiome_preprocessed.h5mu")
ANNOTATED_PATH = os.path.join(PROJECT_DIR, "results/multivi_full_latent_annotated_final.h5ad")
GTF_PATH = os.path.join(PROJECT_DIR, "reference/gencode.v44.annotation.gtf.gz")

RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR = os.path.join(PROJECT_DIR, "results/glue_model")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. Load data
# ============================================================

print("Loading MuData...")
mdata = mu.read_h5mu(DATA_PATH)

rna = mdata.mod["rna"].copy()
atac = mdata.mod["atac"].copy()

print("RNA:")
print(rna)
print("ATAC:")
print(atac)


# ============================================================
# 2. Load final cell type annotation
# ============================================================

print("Loading final cell type annotation...")
annot = sc.read_h5ad(ANNOTATED_PATH)

common_cells = rna.obs_names.intersection(atac.obs_names).intersection(annot.obs_names)

rna = rna[common_cells].copy()
atac = atac[common_cells].copy()
annot = annot[common_cells].copy()

rna.obs["final_cell_type"] = annot.obs["final_cell_type"].astype(str).values
atac.obs["final_cell_type"] = annot.obs["final_cell_type"].astype(str).values

print("Common cells:", len(common_cells))
print("Final cell type counts:")
print(rna.obs["final_cell_type"].value_counts())


# ============================================================
# 3. Use count layers
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

# Select top accessible peaks to reduce memory
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

# Keep annotated and highly variable genes
rna = rna[:, rna.var["chrom"].notna()].copy()
rna = rna[:, rna.var["highly_variable"]].copy()

print("RNA after gene annotation and HVG filtering:")
print(rna)


# ============================================================
# 7. Prepare ATAC peak coordinates
# ============================================================

print("Preparing ATAC peak coordinates...")

# Expected peak format: chr1:10000-10500
# This version is more stable across pandas versions.
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
# 8. Configure datasets for GLUE
# ============================================================

print("Configuring datasets for GLUE...")

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
# 10. Train GLUE
# ============================================================

print("Training GLUE model...")

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
# 11. Encode latent representation
# ============================================================

print("Encoding GLUE latent embeddings...")

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

# Because this is paired multiome data, average RNA-side and ATAC-side embeddings
# to obtain one GLUE latent vector per cell.
glue_latent = (rna.obsm["X_glue"] + atac.obsm["X_glue"]) / 2

glue_adata = ad.AnnData(X=glue_latent)
glue_adata.obs_names = rna.obs_names.copy()
glue_adata.obs["final_cell_type"] = rna.obs["final_cell_type"].astype(str).values

if "rna_leiden" in annot.obs:
    glue_adata.obs["rna_leiden"] = annot.obs["rna_leiden"].astype(str).values

if "atac_leiden" in annot.obs:
    glue_adata.obs["atac_leiden"] = annot.obs["atac_leiden"].astype(str).values

glue_adata.obsm["X_glue"] = glue_latent


# ============================================================
# 12. UMAP and Leiden clustering
# ============================================================

print("Computing UMAP and Leiden...")

sc.pp.neighbors(
    glue_adata,
    use_rep="X_glue",
    n_neighbors=15
)

sc.tl.umap(glue_adata)

sc.tl.leiden(
    glue_adata,
    resolution=1.0,
    key_added="glue_leiden"
)


# ============================================================
# 13. Save results
# ============================================================

print("Saving GLUE outputs...")

glue_adata.write_h5ad(
    os.path.join(RESULT_DIR, "glue_latent.h5ad")
)

pd.DataFrame(
    glue_latent,
    index=glue_adata.obs_names,
    columns=[f"GLUE_{i}" for i in range(glue_latent.shape[1])]
).to_csv(
    os.path.join(RESULT_DIR, "glue_latent.csv")
)

summary_path = os.path.join(RESULT_DIR, "glue_summary.txt")

with open(summary_path, "w") as f:
    f.write("GLUE completed successfully\n")
    f.write(f"Cells: {glue_adata.n_obs}\n")
    f.write(f"Latent dimensions: {glue_latent.shape[1]}\n")
    f.write(f"RNA input shape: {rna.shape}\n")
    f.write(f"ATAC input shape: {atac.shape}\n")
    f.write(f"Guidance graph edges: {guidance.number_of_edges()}\n")
    f.write("\nCell type counts:\n")
    f.write(str(glue_adata.obs["final_cell_type"].value_counts()))


# ============================================================
# 14. Plot figures
# ============================================================

print("Plotting GLUE figures...")

sc.pl.umap(
    glue_adata,
    color="final_cell_type",
    frameon=False,
    legend_loc="right margin",
    title="GLUE latent space: final cell type",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_glue_final_cell_type.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

sc.pl.umap(
    glue_adata,
    color="glue_leiden",
    frameon=False,
    legend_loc="right margin",
    title="GLUE latent space: Leiden clusters",
    show=False
)

plt.savefig(
    os.path.join(FIG_DIR, "umap_glue_leiden.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 15. Finish
# ============================================================

print("Done.")
print("Saved files:")
print(os.path.join(RESULT_DIR, "glue_latent.h5ad"))
print(os.path.join(RESULT_DIR, "glue_latent.csv"))
print(os.path.join(RESULT_DIR, "glue_summary.txt"))
print(os.path.join(FIG_DIR, "umap_glue_final_cell_type.png"))
print(os.path.join(FIG_DIR, "umap_glue_leiden.png"))
