import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# Configuration plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)


### Préparation de données

print('='*80)
print('PARTIE 1 - PREPARATION DES DONNEES')
print('='*80)

fichier_csv = './data/UNSW_NB15_testing-set.csv'

# Charger le fichier (gros fichier, lire en chunks ou limiter)
df = pd.read_csv(fichier_csv)  # Limiter a 50k points pour perf
print(f'   - {len(df):,} points charges')
print(f'   - Colonnes: {list(df.columns)[:8]}...')


print(f'\nStructure des donnees:')


# Etape 2: Nettoyage
print('\n--- NETTOYAGE ---')
print(f'Avant: {len(df):,} points')

# Supprimer NULL
# df = df.dropna()














### Analyse Exploratoire






### Cluster K-mean



### Cluster DBSCAN