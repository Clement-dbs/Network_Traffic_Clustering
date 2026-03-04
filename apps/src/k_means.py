import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

try:
    from .data_loader import load_data, get_tp_features
    from .preprocessing import preprocess_data
except ImportError:
    from data_loader import load_data, get_tp_features
    from preprocessing import preprocess_data

os.makedirs("apps/output/results", exist_ok=True)
os.makedirs("apps/output/figures/k_means", exist_ok=True)
sns.set_style("whitegrid")

donnees = load_data("testing")

variables_principales = get_tp_features(donnees, include_proto=True).columns.tolist()
variables_principales_numeriques = get_tp_features(donnees, include_proto=False).columns.tolist()

os.makedirs("apps/output/results", exist_ok=True)
donnees.head(20).to_csv("apps/output/results/apercu_donnees.csv", index=False)
donnees.describe().to_csv("apps/output/results/statistiques_descriptives.csv")

# résumé global
nb_lignes, nb_colonnes = donnees.shape
colonnes_numeriques = donnees.select_dtypes(include=[np.number]).columns
colonnes_categorielles = donnees.select_dtypes(include=["object"]).columns
colonnes_clustering = [colonne for colonne in colonnes_numeriques if colonne not in ["id", "label"]]

total_manquantes = int(donnees[colonnes_clustering].isnull().sum().sum())

print(f"Dataset: {nb_lignes} lignes, {nb_colonnes} colonnes")
print(f"Variables principales: {variables_principales}\n")

# stats principales
donnees[variables_principales_numeriques].describe().round(2).to_csv("apps/output/results/stats_variables_principales.csv")

# histogrammes (grille 3x3)
figure_hist, axes_hist = plt.subplots(3, 3, figsize=(15, 12))
figure_hist.suptitle("Distribution des variables principales", fontsize=16)
axes_hist = axes_hist.flatten()

for index, colonne in enumerate(variables_principales_numeriques[:9]):
    donnees[colonne].hist(bins=50, ax=axes_hist[index], edgecolor="black", alpha=0.7)

for index in range(len(variables_principales_numeriques[:9]), 9):
    axes_hist[index].axis("off")

plt.tight_layout()
plt.savefig("apps/output/figures/k_means/distribution_variables_principales.png", dpi=300, bbox_inches="tight")

# boxplots (grille 3x3)
figure_box, axes_box = plt.subplots(3, 3, figsize=(15, 12))
figure_box.suptitle("Boxplots des variables principales", fontsize=16)
axes_box = axes_box.flatten()

for index, colonne in enumerate(variables_principales_numeriques[:9]):
    donnees.boxplot(column=colonne, ax=axes_box[index])

for index in range(len(variables_principales_numeriques[:9]), 9):
    axes_box[index].axis("off")

plt.tight_layout()
plt.savefig("apps/output/figures/k_means/boxplots_variables_principales.png", dpi=300, bbox_inches="tight")

# rapport outliers IQR
nb_lignes_total = len(donnees)
lignes_rapport = []

for colonne in variables_principales_numeriques:
    quartile_1 = donnees[colonne].quantile(0.25)
    quartile_3 = donnees[colonne].quantile(0.75)
    ecart_interquartile = quartile_3 - quartile_1
    limite_basse = quartile_1 - 1.5 * ecart_interquartile
    limite_haute = quartile_3 + 1.5 * ecart_interquartile

    nb_outliers = int(((donnees[colonne] < limite_basse) | (donnees[colonne] > limite_haute)).sum())
    nb_zeros = int((donnees[colonne] == 0).sum())
    nb_negatifs = int((donnees[colonne] < 0).sum())

    lignes_rapport.append([
        colonne,
        nb_outliers,
        nb_outliers / nb_lignes_total * 100,
        nb_zeros,
        nb_negatifs,
        limite_basse,
        limite_haute,
    ])

rapport_outliers = pd.DataFrame(
    lignes_rapport,
    columns=["var", "outliers", "outliers_%", "zeros", "negatifs", "limite_inf", "limite_sup"],
).sort_values("outliers", ascending=False)

rapport_outliers.to_csv("apps/output/results/outliers_iqr.csv", index=False)

# analyse du coude (elbow method)
donnees_normalisees, _ = preprocess_data(donnees, include_proto=True)

inertias = []
range_k = range(1, 11)

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(donnees_normalisees)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range_k, inertias, "bo-", linewidth=2, markersize=8)
plt.xlabel("Nombre de clusters (k)", fontsize=12)
plt.ylabel("Inertie", fontsize=12)
plt.title("Analyse du coude (Elbow Method)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("apps/output/figures/k_means/coude_kmeans.png", dpi=300, bbox_inches="tight")

print("✓ Fichiers CSV générés dans apps/output/results/")
print("✓ Fichiers PNG générés dans apps/output/figures/k_means/")
