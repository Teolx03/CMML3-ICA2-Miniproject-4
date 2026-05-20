import muon as mu
import scanpy as sc
import matplotlib.pyplot as plt

# Load data
mdata = mu.read_10x_h5("10k_PBMC_Multiome_nextgem_Chromium_X_filtered_feature_bc_matrix.h5")

# Make names unique
mdata.var_names_make_unique()
mdata.mod["rna"].var_names_make_unique()
mdata.mod["atac"].var_names_make_unique()

# Extract RNA and ATAC
rna = mdata.mod["rna"]
atac = mdata.mod["atac"]

# Save raw counts
rna.layers["counts"] = rna.X.copy()
atac.layers["counts"] = atac.X.copy()

# RNA QC
rna.var["mt"] = rna.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    rna,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True
)

# ATAC QC
sc.pp.calculate_qc_metrics(
    atac,
    percent_top=None,
    log1p=False,
    inplace=True
)

print("RNA shape:", rna.shape)
print("ATAC shape:", atac.shape)

print("\nRNA QC summary:")
print(rna.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())

print("\nATAC QC summary:")
print(atac.obs[["total_counts", "n_genes_by_counts"]].describe())    # number of detected ATAC peak features per cell.


# RNA QC plots
sc.pl.violin(
    rna,
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True
)

# ATAC QC plots
sc.pl.violin(
    atac,
    ["total_counts", "n_genes_by_counts"],
    jitter=0.4,
    multi_panel=True
)



# -----------------------------
# QC filtering
# -----------------------------

# RNA filtering mask
rna_mask = (
    (rna.obs["n_genes_by_counts"] > 500) &
    (rna.obs["n_genes_by_counts"] < 6000) &
    (rna.obs["total_counts"] > 500) &
    (rna.obs["pct_counts_mt"] < 20)
)

# ATAC filtering mask
atac_mask = (
    (atac.obs["n_genes_by_counts"] > 1000) &
    (atac.obs["n_genes_by_counts"] < 30000) &
    (atac.obs["total_counts"] > 1000) &
    (atac.obs["total_counts"] < 100000)
)

print("Cells before filtering:", rna.n_obs)

print("RNA pass QC:", rna_mask.sum())
print("ATAC pass QC:", atac_mask.sum())

# Keep only cells passing BOTH RNA and ATAC QC
common_mask = rna_mask & atac_mask

rna_qc = rna[common_mask].copy()
atac_qc = atac[common_mask].copy()

print("Cells after joint RNA+ATAC QC:", rna_qc.n_obs)

print("RNA after QC:", rna_qc.shape)
print("ATAC after QC:", atac_qc.shape)


# Filter genes detected in very few cells
sc.pp.filter_genes(rna_qc, min_cells=3)

# Filter ATAC peaks detected in very few cells
sc.pp.filter_genes(atac_qc, min_cells=10)

print("RNA after gene filtering:", rna_qc.shape)
print("ATAC after peak filtering:", atac_qc.shape)

# Save raw counts again after QC
rna_qc.layers["counts"] = rna_qc.X.copy()
atac_qc.layers["counts"] = atac_qc.X.copy()

# Put back into MuData
mdata.mod["rna"] = rna_qc
mdata.mod["atac"] = atac_qc
mdata.update()

# Save the QC-filtered object
mdata.write("pbmc_multiome_qc.h5mu")

print("Saved QC-filtered MuData object.")



# -----------------------------
# RNA preprocessing
# -----------------------------

rna_qc.layers["counts"] = rna_qc.X.copy()

sc.pp.normalize_total(rna_qc, target_sum=1e4)
sc.pp.log1p(rna_qc)

sc.pp.highly_variable_genes(
    rna_qc,
    n_top_genes=3000,
    flavor="seurat"
)

print("Number of highly variable genes:")
print(rna_qc.var["highly_variable"].sum())

# Use HVGs for PCA/UMAP
rna_hvg = rna_qc[:, rna_qc.var["highly_variable"]].copy()

sc.pp.scale(rna_hvg, max_value=10)
sc.tl.pca(rna_hvg, n_comps=50, svd_solver="arpack")

sc.pp.neighbors(rna_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(rna_hvg)
sc.tl.leiden(rna_hvg, resolution=0.5)

# Copy UMAP and clusters back to full RNA object
rna_qc.obsm["X_umap"] = rna_hvg.obsm["X_umap"]
rna_qc.obs["rna_leiden"] = rna_hvg.obs["leiden"].copy()

# Plot RNA-only clusters
sc.pl.umap(rna_qc, color=["rna_leiden"])

# Check marker genes
pbmc_markers = [
    "CD3D", "CD4", "CD8A",
    "MS4A1", "CD79A",
    "NKG7", "GNLY",
    "LYZ", "LST1", "S100A8", "S100A9",
    "FCGR3A",
    "PPBP", "PF4"
]

# Plot marker genes
available_markers = [g for g in pbmc_markers if g in rna_qc.var_names]

print("Available markers:", available_markers)

sc.pl.umap(
    rna_qc,
    color=available_markers,
    use_raw=False,
    cmap="viridis",
    ncols=4
)





# -----------------------------
# ATAC preprocessing
# -----------------------------

# Save raw ATAC counts for MultiVI
atac_qc.layers["counts"] = atac_qc.X.copy()

# ATAC data are usually very sparse
print("ATAC before processing:", atac_qc.shape)

# TF-IDF normalization  ()
mu.atac.pp.tfidf(atac_qc)

# LSI dimensionality reduction
mu.atac.tl.lsi(atac_qc, n_comps=50)

print("ATAC obsm keys:", atac_qc.obsm.keys())
print("ATAC LSI shape:", atac_qc.obsm["X_lsi"].shape)

# Usually remove the first LSI component because it often captures sequencing depth
sc.pp.neighbors(
    atac_qc,
    use_rep="X_lsi",
    n_neighbors=15,
    n_pcs=30
)

sc.tl.umap(atac_qc)
sc.tl.leiden(atac_qc, resolution=0.5)

# Rename Leiden result for clarity
atac_qc.obs["atac_leiden"] = atac_qc.obs["leiden"].copy()

# Plot ATAC-only UMAP
sc.pl.umap(
    atac_qc,
    color="atac_leiden",
    legend_loc="on data"
)


# -----------------------------
# Save preprocessed MuData object
# -----------------------------

mdata.mod["rna"] = rna_qc
mdata.mod["atac"] = atac_qc
mdata.update()

print("Final preprocessed object:")
print(mdata)
print("RNA:", mdata.mod["rna"].shape)
print("ATAC:", mdata.mod["atac"].shape)

mdata.write("pbmc_multiome_preprocessed.h5mu")

print("Saved as pbmc_multiome_preprocessed.h5mu")