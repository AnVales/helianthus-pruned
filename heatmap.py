import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. PCA + CLUSTERS
# =========================

df = pd.read_csv("pca_result.eigenvec", sep=r"\s+", header=None)

df.columns = [
    "FID","IID","PC1","PC2","PC3","PC4","PC5",
    "PC6","PC7","PC8","PC9","PC10"
]

clusters = pd.read_csv("clusters.tsv", sep=r"\s+", header=None)
clusters.columns = ["FID","IID","cluster"]

# asegurar tipo correcto (evita errores merge)
df["FID"] = df["FID"].astype(str)
df["IID"] = df["IID"].astype(str)
clusters["FID"] = clusters["FID"].astype(str)
clusters["IID"] = clusters["IID"].astype(str)

df = df.merge(clusters, on=["FID","IID"])

pc_cols = ["PC1","PC2","PC3"]

# =========================
# 2. HEATMAP MEDIAS PCA
# =========================

cluster_means = df.groupby("cluster")[pc_cols].mean()

plt.figure(figsize=(7,5))
sns.heatmap(cluster_means, annot=True, cmap="viridis")

plt.title("Medias de PCs por cluster")
plt.xlabel("Componentes principales")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

# =========================
# 3. HEATMAP INDIVIDUOS
# =========================

df_sorted = df.sort_values("cluster")

plt.figure(figsize=(10,6))
sns.heatmap(df_sorted[pc_cols].T, cmap="coolwarm", cbar=True)

plt.title("Individuos ordenados por cluster")
plt.xlabel("Individuos")
plt.ylabel("PCs")
plt.tight_layout()
plt.show()

# =========================
# 4. HISTOGRAMA FST (SNP LEVEL)
# =========================

fst = pd.read_csv("fst_results.fst", sep=r"\s+")

plt.figure(figsize=(7,5))
plt.hist(fst["FST"], bins=50, color="steelblue")

plt.title("Distribución FST por SNP")
plt.xlabel("FST")
plt.ylabel("Número de SNPs")
plt.tight_layout()
plt.show()


# =========================
# 5. HEATMAP FST PAIRWISE 
# =========================

import re

pairs = []

with open("clean_fst.txt") as f:
    for line in f:

        # SOLO aceptar líneas correctas
        if "fst_" not in line or "Fst" not in line:
            continue

        match = re.search(
            r"fst_(\d+)_(\d+).*?Fst estimate:\s*([0-9.]+)",
            line
        )

        if match:
            c1 = int(match.group(1))
            c2 = int(match.group(2))
            fst = float(match.group(3))
            pairs.append([c1, c2, fst])

df = pd.DataFrame(pairs, columns=["c1", "c2", "fst"])

n = max(df["c1"].max(), df["c2"].max()) + 1
matrix = np.zeros((n, n))

for _, row in df.iterrows():
    i, j, v = int(row["c1"]), int(row["c2"]), row["fst"]
    matrix[i, j] = v
    matrix[j, i] = v

np.fill_diagonal(matrix, 0)

plt.figure(figsize=(6,5))
plt.imshow(matrix, cmap="viridis")
plt.colorbar(label="FST")
plt.xticks(range(n))
plt.yticks(range(n))
plt.xlabel("clusters")
plt.ylabel("clusters")
plt.title("FST entre clusters")
plt.tight_layout()
plt.show()

# =========================
# 6. Dendrograma 
# =========================
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# Convertir FST en distancia
dist_matrix = matrix

# Clustering jerárquico
condensed = squareform(dist_matrix)
Z = linkage(condensed, method="average")
