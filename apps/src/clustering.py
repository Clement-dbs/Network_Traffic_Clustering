from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Backend non-interactif (pour serveurs/CI). À mettre AVANT tout affichage.
matplotlib.use("Agg")

try:
    from .data_loader import load_data
    from .preprocessing import preprocess_data
except ImportError:
    from data_loader import load_data
    from preprocessing import preprocess_data


FIGURE_ROOT = Path("apps/output/figures")


def figure_path(method_name: str, filename: str) -> Path:
    path = (FIGURE_ROOT / method_name / filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dense_matrix(X):
    return X.toarray() if hasattr(X, "toarray") else X


def project_for_plot(X, method="pca"):
    X = ensure_dense_matrix(X)
    if X.shape[1] <= 2:
        return X[:, :2]
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(X)
    return X[:, :2]


def save_cluster_figure(X, labels, title, output_path: Path, plot_method="pca"):
    X = ensure_dense_matrix(X)
    X_2d = project_for_plot(X, method=plot_method)

    labels = np.asarray(labels)
    if labels.min() == 1:
        labels = labels - 1

    plt.figure(figsize=(9, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=10, cmap="tab10", alpha=0.85)
    plt.title(title)
    plt.xlabel("Component 1" if plot_method == "pca" else "Feature 1")
    plt.ylabel("Component 2" if plot_method == "pca" else "Feature 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)


def save_dendrogram(Z, output_path: Path, p=6):
    plt.figure(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=p, no_labels=True, color_threshold=0)
    plt.title("Dendrogram (Ward)")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return str(output_path)


def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= X.shape[0]:
        return None
    return float(silhouette_score(X, labels))


def main():

    df = load_data("testing").head(1000)
    X, _ = preprocess_data(df, include_proto=True)
    X = ensure_dense_matrix(X)

    # 2)StandardScaler pour Ward/Euclidien 
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # 3) Sampling
    sample_size = min(2000, X.shape[0])
    rng = np.random.default_rng(42)
    indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    X_sample = X[indices]

    # 4) Linkage (arbre hiérarchique)
    Z = linkage(X_sample, method="ward")
    dendro_path = save_dendrogram(
        Z,
        figure_path("agglomerative", "dendrogram_ward.png"),
        p=6,
    )

    # ---- Méthode 1 : fcluster en fixant K (criterion="maxclust")
    K = 6
    labels_k = fcluster(Z, t=K, criterion="maxclust")
    sil_k = safe_silhouette(X_sample, labels_k)
    k_path = save_cluster_figure(
        X_sample,
        labels_k,
        title=f"fcluster (maxclust) - K={K} - silhouette={sil_k if sil_k is not None else 'NA'}",
        output_path=figure_path("agglomerative", "agglomerative_fcluster_k.png"),
        plot_method="pca",
    )

    # ---- Méthode 2 : fcluster en coupant à une hauteur (distance)
    cut_height = float(np.percentile(Z[:, 2], 75))
    labels_h = fcluster(Z, t=cut_height, criterion="distance")
    sil_h = safe_silhouette(X_sample, labels_h)
    h_path = save_cluster_figure(
        X_sample,
        labels_h,
        title=f"fcluster (distance) - cut_height={cut_height:.2f} - silhouette={sil_h if sil_h is not None else 'NA'}",
        output_path=figure_path("agglomerative", "agglomerative_fcluster_height.png"),
        plot_method="pca",
    )

    # ---- Méthode 3 : scikit-learn AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=K, linkage="ward")
    labels_model = model.fit_predict(X_sample)
    sil_model = safe_silhouette(X_sample, labels_model)
    model_path = save_cluster_figure(
        X_sample,
        labels_model,
        title=f"AgglomerativeClustering (sklearn) - K={K} - silhouette={sil_model if sil_model is not None else 'NA'}",
        output_path=figure_path("agglomerative", "agglomerative_model_k.png"),
        plot_method="pca",
    )

    print("Dendrogram saved to:", dendro_path)
    print("fcluster (K) saved to:", k_path)
    print("fcluster (height) saved to:", h_path)
    print("AgglomerativeClustering saved to:", model_path)
    print("Sample size:", sample_size)
    print("Silhouette (fcluster K):", sil_k)
    print("Silhouette (fcluster height):", sil_h)
    print("Silhouette (sklearn):", sil_model)


if __name__ == "__main__":
    main()