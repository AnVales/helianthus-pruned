import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D


# =========================
# 1. CARGA PCA
# =========================
df = pd.read_csv("pca_result.eigenvec", delim_whitespace=True, header=None)

df.columns = [
    "FID","IID","PC1","PC2","PC3","PC4","PC5",
    "PC6","PC7","PC8","PC9","PC10"
]

X = df[["PC1", "PC2", "PC3"]]


# =========================
# 2. VARIANZA EXPLICADA (eigenval)
# =========================
eigenvals = np.loadtxt("pca_result.eigenval")

explained = eigenvals / np.sum(eigenvals)

print("\nVarianza explicada por PC:")
for i, v in enumerate(explained[:10], start=1):
    print(f"PC{i}: {v*100:.2f}%")


# Scree plot
plt.figure()
plt.bar(range(1, 11), explained[:10])
plt.xlabel("Principal Component")
plt.ylabel("Varianza explicada")
plt.title("Scree plot (varianza explicada por PC)")
plt.show()


# =========================
# 2.5 ELBOW METHOD (INERCIA)
# =========================
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(list(K_range), inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inercia (SSE)")
plt.title("Elbow Method")
plt.show()


# =========================
# 3. BUSCAR MEJOR K (SILHOUETTE)
# =========================
best_k = None
best_score = -1
best_model = None

K_range = range(2, 11)
scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)
    scores.append(score)

    if score > best_score:
        best_k = k
        best_score = score
        best_model = kmeans


print(f"\nMejor K: {best_k}")
print(f"Silhouette score: {best_score:.3f}")


# curva silhouette
plt.figure()
plt.plot(list(K_range), scores, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.title("Selección automática de K")
plt.show()


# =========================
# 4. CLUSTER FINAL
# =========================
df["cluster"] = best_model.labels_


# =========================
# 5. GUARDAR RESULTADOS
# =========================
df[["FID","IID","cluster"]].to_csv("clusters.tsv", sep="\t", index=False)


# =========================
# 6. FUNCIÓN PLOT
# =========================
def plot(x, y, title):
    plt.figure()

    for c in df["cluster"].unique():
        subset = df[df["cluster"] == c]
        plt.scatter(subset[x], subset[y], label=f"Cluster {c}", alpha=0.6)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.show()


# =========================
# 7. GRÁFICOS PCA 2D
# =========================
plot("PC1", "PC2", "PC1 vs PC2")
plot("PC1", "PC3", "PC1 vs PC3")
plot("PC2", "PC3", "PC2 vs PC3")


# =========================
# 8. PCA 3D
# =========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c in df["cluster"].unique():
    subset = df[df["cluster"] == c]
    ax.scatter(
        subset["PC1"],
        subset["PC2"],
        subset["PC3"],
        label=f"Cluster {c}",
        alpha=0.7
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA 3D + clustering")

plt.legend()
plt.show()