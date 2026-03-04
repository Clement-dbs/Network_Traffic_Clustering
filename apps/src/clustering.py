from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from data_loader import load_data
from preprocessing import preprocess_data


def save_cluster_figure(X, labels, output_path="apps/output/figures/agglomerative_clusters.png"):
    if X.shape[1] < 2:
        raise ValueError("X doit avoir au moins 2 colonnes pour tracer la figure.")

    X_2d = X[:, :2]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=8, cmap="tab10", alpha=0.8)
    plt.title("Agglomerative Clustering (2 premieres variables)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    return str(output_file)


if __name__ == "__main__":
    df = load_data("testing").head(5000)
    X, _ = preprocess_data(df, include_proto=True)

    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    print("k:", 3)
    print("Silhouette:", round(score, 4))
    print("Nb labels:", len(labels))
    saved_path = save_cluster_figure(X=X, labels=labels)
    print("Figure saved:", saved_path)
