from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

try:
    from .data_loader import load_data
    from .preprocessing import preprocess_data
except ImportError:
    from data_loader import load_data
    from preprocessing import preprocess_data


def save_cluster_figure(X, labels, title, output_path):
    if X.shape[1] < 2:
        print("Pas assez de dimensions pour faire une figure.")
        return
def ensure_dense_matrix(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X
def project_for_plot(X, method="pca"):
    if X.shape[1] <= 2:
        return X[:, :2]
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(X)
    return X[:, :2]

def save_cluster_figure(X, labels, title, output_path, plot_method="pca"):
    X = ensure_dense_matrix(X)
    X_2d = project_for_plot(X, method=plot_method)

    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=8, cmap="tab10", alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1" if plot_method == "pca" else "Feature 1")
    plt.ylabel("Component 2" if plot_method == "pca" else "Feature 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    return str(output_file)

def save_dendrogram(Z, output_path, p=6):
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=p, no_labels=True, color_threshold=0)
    plt.title("Dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    return str(output_file)

def save_silhouette(X, labels):
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= X.shape[0]:
        return None
    return float(silhouette_score(X, labels))
if __name__ == "__main__":
    df = load_data("testing").head(1000)
    X, scaler = preprocess_data(df, include_proto=True)
    X = ensure_dense_matrix(X)

    sample_size = min(2000, X.shape[0])
    rng = np.random.default_rng(42)
    indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    X_sample = X[indices]

    Z = linkage(X_sample, method="ward")
    dendro_path = save_dendrogram(Z, "apps/output/figures/dendrogram_ward.png", p=6)


# for k in range(2,10):
#     model = AgglomerativeClustering(n_clusters=k, linkage="ward")
#     labels = model.fit_predict(X_sample)
#     score = silhouette_score(X_sample, labels)
#     scores.append(score)
#     print("K =",k,"score =",score)

# plt.plot(range(2,10), scores)
# plt.xlabel("Nombre de clusters")
# plt.ylabel("Silhouette score")
# plt.show()
    K = 6
    cut_height = float(np.percentile(Z[:, 2],75))
    labels_height = fcluster(Z, cut_height, criterion="distance")
    height_path = save_cluster_figure(
        X_sample,
        labels_height,
        title=f"fcluster - coupe a hauteur={cut_height:.2f}",
        output_path="apps/output/figures/agglomerative_fcluster_height.png",
        plot_method="pca",
    )

    model = AgglomerativeClustering(n_clusters=K, linkage="ward")
    labels_model = model.fit_predict(X_sample)
    sil = save_silhouette(X_sample, labels_model)
    model_path = save_cluster_figure(
        X_sample,
        labels_model,
        title=f"AgglomerativeClustering - K={K} - silhouette={sil:.3f}",
        output_path="apps/output/figures/agglomerative_model_k.png",
        plot_method="pca",
    )
    print("Dendrogram saved to:", dendro_path)
    print("fcluster (height) saved to:", height_path)
    print("AgglomerativeClustering saved to:", model_path) 
    print("Sample size:", sample_size)
    print("Silhouette score:", sil)



   



