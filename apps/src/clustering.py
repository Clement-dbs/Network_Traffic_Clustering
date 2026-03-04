from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from data_loader import load_data
from preprocessing import preprocess_data


def run_agglomerative(X, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)
    return labels, model


def choose_k_with_silhouette(X, min_k=2, max_k=8):
    scores = {}
    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        labels, _ = run_agglomerative(X, n_clusters=k)
        score = silhouette_score(X, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, scores


if __name__ == "__main__":
    df = load_data("testing").head(5000)
    X, _ = preprocess_data(df, include_proto=True)

    best_k, best_score, all_scores = choose_k_with_silhouette(X, min_k=2, max_k=6)
    labels, _ = run_agglomerative(X, n_clusters=best_k)

    print("Best k:", best_k)
    print("Best silhouette:", round(best_score, 4))
    print("Scores:", all_scores)
    print("Nb labels:", len(labels))
