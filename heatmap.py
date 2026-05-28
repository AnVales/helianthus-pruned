import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =========================
# 1. CARGAR DATOS
# =========================

df = pd.read_csv("pca_result.eigenvec", delim_whitespace=True, header=None)
df.columns = [
    "FID","IID","PC1","PC2","PC3","PC4","PC5",
    "PC6","PC7","PC8","PC9","PC10"
]

clusters = pd.read_csv("clusters.tsv", sep="\t")

# unir clusters con PCA
df = df.merge(clusters, on=["FID", "IID"])

# preparar matriz de FST
fst = pd.read_csv("fst_results.fst", sep=r"\s+")

# preparar matriz de FST (si es por cluster)
fst = pd.read_csv("fst_pairwise.fst", sep=r"\s+")

# si es por SNP → agregamos promedio
mean_fst = fst["FST"].mean()

# reconstruir matriz aproximada (si no hay pares explícitos)
clusters = [0,1,2,3]

mat = pd.DataFrame(
    np.full((4,4), mean_fst),
    index=clusters,
    columns=clusters
)

# =========================
# 2. Graficar FST entre clusters (heatmap)
# =========================

plt.figure(figsize=(8,5))

plt.hist(fst["FST"], bins=50)

plt.xlabel("FST")
plt.ylabel("Número de SNPs")
plt.title("Histograma de FST por SNP")

plt.show()

# =========================
# 2. HEATMAP (MEDIAS POR CLUSTER)
# =========================

pc_cols = ["PC1", "PC2", "PC3"]

cluster_means = df.groupby("cluster")[pc_cols].mean()

plt.figure(figsize=(8,5))
sns.heatmap(cluster_means, annot=True, cmap="viridis")

plt.title("Heatmap: medias de PCs por cluster")
plt.xlabel("Principal Components")
plt.ylabel("Cluster")

plt.tight_layout()
plt.show()


# =========================
# 3. HEATMAP INDIVIDUAL 
# =========================

df_sorted = df.sort_values("cluster")

plt.figure(figsize=(10,6))

sns.heatmap(
    df_sorted[pc_cols].T,
    cmap="coolwarm",
    cbar=True
)

plt.title("Heatmap de individuos (ordenados por cluster)")
plt.xlabel("Individuos")
plt.ylabel("PCs")

plt.tight_layout()
plt.show()

# =========================
# 3. HEATMAP FST ENTRE CLUSTERS
# =========================
# =========================
# 1. cargar FST por SNP
# =========================
fst = pd.read_csv("fst_results.fst", sep=r"\s+")

# =========================
# 2. cargar clusters
# =========================
clusters = pd.read_csv("clusters.tsv", sep=r"\s+", header=None)
clusters.columns = ["FID","IID","cluster"]

# =========================
# 3. como PLINK NO da pares explícitos:
#    usamos FST medio como proxy por cluster-pair
# =========================

unique_clusters = sorted(clusters["cluster"].unique())
n = len(unique_clusters)

mat = np.zeros((n, n))

# rellenamos con media global de FST
# (esto es aproximación si no hay pairwise explícito)
global_mean = fst["FST"].mean()

for i in range(n):
    for j in range(n):
        if i == j:
            mat[i, j] = 0
        else:
            mat[i, j] = global_mean

# simetrizar (opcional)
mat = (mat + mat.T) / 2

# =========================
# 4. heatmap
# =========================
plt.figure(figsize=(6,5))

sns.heatmap(
    mat,
    annot=True,
    cmap="Reds",
    xticklabels=unique_clusters,
    yticklabels=unique_clusters
)

plt.title("FST entre clusters (reconstruido)")
plt.xlabel("Cluster")
plt.ylabel("Cluster")

plt.tight_layout()
plt.show()