import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from pathlib import Path
from data_loader import load_data, get_tp_features
from preprocessing import preprocess_data

### Préparation de données

print('='*80)
print('PARTIE 1 - PREPARATION DES DONNEES')
print('='*80)

# Charger le fichier
df = load_data("testing").head(20000)

print(f'   - {len(df):,} points')
print(f'   - Colonnes: {list(df.columns)}')
print(df.describe())

# Garder les colonnes importantes
# * dur : durée de la connexion
# * proto : protocole réseau
# * sbytes : nombre d’octets envoyés par la source
# * dbytes : nombre d’octets envoyés par la destination
# * sttl : TTL de la source
# * dttl : TTL de la destination
# * spkts : nombre de paquets source
# * dpkts : nombre de paquets destination
# * rate : taux de transfert
# features = get_tp_features(df, include_proto=False).columns.tolist()

features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate', 'sttl', 'dttl']
df = df[features] 
df = df.head(20000)

# Etape 2: Nettoyage
print('='*80)
print('PARTIE 2 - NETTOYAGE')
print('='*80)
print(f'Avant nettoyage : {len(df):,} points')

# Traiter les variables null
print(df.isna().sum())
df = df.dropna()
print(f'Après nettoyage : {len(df):,} points')


# Cluster DBSCAN

print('='*80)
print('PARTIE 3 - PREPROCESSING')
print('='*80)

## Preprocessing
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["string"]).columns

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scaler',  StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])


print('\n' + '='*80)
print('PARTIE 4 - METHODE k-DISTANCE POUR CHOISIR EPSILON')
print('='*80)

## Trouver les NN 
X_scaled = preprocessor.fit_transform(df)
min_samples = 2 * X_scaled.shape[1]

k = min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

distances_k = distances[:, k-1]
distances_k_sorted = np.sort(distances_k)[::-1]

# Courbe k-distance
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(distances_k_sorted, 'b-', linewidth=1.5, label='k-distance')
ax.set_xlabel('Points triés (index)')
ax.set_ylabel(f'Distance au {k}e voisin')
ax.set_title(f'Méthode k-distance pour choisir epsilon\n(k={k}, metric=euclidean)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Identifier le coude
diff1 = np.diff(distances_k_sorted)
diff2 = np.diff(diff1)
coude_idx = np.argmax(diff2) + 1
epsilon = coude_idx

# Marquer le coude
ax.axhline(y=epsilon, color='red', linestyle='--', linewidth=2,
           label=f'Coude identifie (~{epsilon:.3f})')
ax.plot(coude_idx, epsilon, 'ro', markersize=8, label='Coude')
ax.legend(fontsize=11)
plt.tight_layout()
Path("apps/output/figures/dbscan").mkdir(parents=True, exist_ok=True)
plt.savefig("apps/output/figures/dbscan/k-distance.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


### Application DBSCAN

print('\n' + '='*80)
print('PARTIE 5 - APPLICATION DBSCAN')
print('='*80)


dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')
dbscan.fit(X_scaled)
labels = dbscan.labels_

# Ajouter au dataframe
df['cluster'] = labels
df['cluster_name'] = df['cluster'].apply(lambda x: 'BRUIT' if x == -1 else f'Cluster {x}')

# Statistiques
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_bruit = len(labels[labels == -1])

for cluster_id in sorted(set(labels)):
    mask = labels == cluster_id
    print(f'Cluster {cluster_id}: {mask.sum()} points')

for cluster_id in sorted(set(labels)):
    if cluster_id == -1:
        continue
    mask = labels == cluster_id
    cluster_center = df.loc[mask, features].mean()
    print(f'Cluster {cluster_id} :')
    for col in features:
        print(f'   {col}: {cluster_center[col]:.2f}')

### Visualisation 

print('\n' + '='*80)
print('PARTIE 6 - VISUALISATION')
print('='*80)

x_col = 'sbytes'
y_col = 'dbytes'

fig, ax = plt.subplots(figsize=(14, 10))

colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

for cluster_id in sorted(set(labels)):
    mask = labels == cluster_id
    if cluster_id == -1:
        ax.scatter(df.loc[mask, x_col],
                   df.loc[mask, y_col],
                   marker='x', color='red', s=100,
                   label=f'Bruit ({mask.sum()} pts)', alpha=0.6)
    else:
        color = colors[cluster_id % len(colors)]
        ax.scatter(df.loc[mask, x_col],
                   df.loc[mask, y_col],
                   color=color, s=50,
                   label=f'Cluster {cluster_id} ({mask.sum()} pts)',
                   alpha=0.7, edgecolors='black', linewidth=0.3)

for cluster_id in sorted(set(labels)):
    if cluster_id == -1:
        continue
    mask = labels == cluster_id
    center_x = df.loc[mask, x_col].mean()
    center_y = df.loc[mask, y_col].mean()
    ax.plot(center_x, center_y, 'k*', markersize=20, markeredgecolor='yellow', markeredgewidth=2)

ax.set_xlabel(x_col, fontsize=12)
ax.set_ylabel(y_col, fontsize=12)
ax.set_title(f'Clusters identifiés par DBSCAN\nN={n_clusters} clusters, bruit={n_bruit} pts', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
Path("apps/output/figures/dbscan").mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(str(OUTPUT_FIGURES_DBSCAN / "clusters.png"), dpi=300, bbox_inches='tight')
# plt.show() - Commenté pour éviter les blocages en mode headless