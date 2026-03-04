import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("data", exist_ok=True)
sns.set_style("whitegrid")

donnees = pd.read_csv("data/UNSW_NB15_testing-set.csv")

variables_principales = ["dur", "sbytes", "dbytes", "spkts", "dpkts", "rate", "sttl", "dttl"]

# exports de base
donnees.head(20).to_csv("data/apercu_donnees.csv", index=False)
donnees.describe().to_csv("data/statistiques_descriptives.csv")

# résumé global
nb_lignes, nb_colonnes = donnees.shape
colonnes_numeriques = donnees.select_dtypes(include=[np.number]).columns
colonnes_categorielles = donnees.select_dtypes(include=["object"]).columns
colonnes_clustering = [colonne for colonne in colonnes_numeriques if colonne not in ["id", "label"]]

total_manquantes = int(donnees[colonnes_clustering].isnull().sum().sum())

print(f"Dataset: {nb_lignes} lignes, {nb_colonnes} colonnes")
print(f"Variables principales: {variables_principales}\n")

# stats principales
donnees[variables_principales].describe().round(2).to_csv("data/stats_variables_principales.csv")

# histogrammes (grille 3x3)
figure_hist, axes_hist = plt.subplots(3, 3, figsize=(15, 12))
figure_hist.suptitle("Distribution des variables principales", fontsize=16)
axes_hist = axes_hist.flatten()

for index, colonne in enumerate(variables_principales[:9]):
    donnees[colonne].hist(bins=50, ax=axes_hist[index], edgecolor="black", alpha=0.7)

for index in range(len(variables_principales[:9]), 9):
    axes_hist[index].axis("off")

plt.tight_layout()
plt.savefig("data/distribution_variables_principales.png", dpi=300, bbox_inches="tight")

# boxplots (grille 3x3)
figure_box, axes_box = plt.subplots(3, 3, figsize=(15, 12))
figure_box.suptitle("Boxplots des variables principales", fontsize=16)
axes_box = axes_box.flatten()

for index, colonne in enumerate(variables_principales[:9]):
    donnees.boxplot(column=colonne, ax=axes_box[index])

for index in range(len(variables_principales[:9]), 9):
    axes_box[index].axis("off")

plt.tight_layout()
plt.savefig("data/boxplots_variables_principales.png", dpi=300, bbox_inches="tight")

# rapport outliers IQR
nb_lignes_total = len(donnees)
lignes_rapport = []

for colonne in variables_principales:
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

rapport_outliers.to_csv("data/outliers_iqr.csv", index=False)

# matrice de corrélation
correlations = donnees[variables_principales].corr()
correlations.to_csv("data/matrice_correlation.csv")

plt.figure(figsize=(14, 10))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=1)
plt.title("Matrice de corrélation (variables principales)")
plt.tight_layout()
plt.savefig("data/matrice_correlation.png", dpi=300, bbox_inches="tight")
