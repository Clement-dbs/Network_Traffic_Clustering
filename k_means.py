import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('data', exist_ok=True)

# Configuration pandas pour améliorer la lisibilité dans le terminal
# Raison : Le dataset a 44 colonnes, afficher tout serait illisible
pd.set_option('display.max_columns', 10)      # Limiter colonnes affichées
pd.set_option('display.width', 120)           # Largeur optimale pour terminal
pd.set_option('display.max_rows', 20)         # Limiter lignes affichées
pd.set_option('display.precision', 2)         # 2 décimales suffisent
sns.set_style('whitegrid')                     # Style graphiques élégant


print("PARTIE 1 - ANALYSE EXPLORATOIRE")
print("="*30)


# ============================================================================
# 1. CHARGEMENT ET COMPRÉHENSION DE LA STRUCTURE DU DATASET
# ============================================================================

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Charger les données et comprendre leur structure
# CE QU'ON FAIT : Lecture CSV et affichage des dimensions
# POURQUOI : Première étape obligatoire avant toute analyse
# ----------------------------------------------------------------------------

print("\n1.1 Dimensions du dataset:")

# Charger le fichier CSV dans un DataFrame
# Raison : pandas offre des outils puissants pour l'analyse de données tabulaires
donnees_reseau = pd.read_csv('data/UNSW_NB15_testing-set.csv')

nombre_connexions = donnees_reseau.shape[0]
nombre_variables = donnees_reseau.shape[1]

print(f"   - Nombre de lignes (connexions réseau): {nombre_connexions}")
print(f"   - Nombre de colonnes (variables): {nombre_variables}")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Sauvegarder les données brutes pour analyse externe
# CE QU'ON FAIT : Export CSV des premières lignes et statistiques
# POURQUOI : Le terminal n'affiche pas tout, les CSV permettent une consultation
#            détaillée dans Excel ou autre outil
# ----------------------------------------------------------------------------

print("\n   Sauvegarde des données complètes pour consultation détaillée...")

# Sauvegarder un aperçu des données (20 premières lignes)
# Raison : Permet d'examiner toutes les colonnes sans surcharger le terminal
donnees_reseau.head(20).to_csv('data/apercu_donnees.csv', index=False)

# Sauvegarder les statistiques descriptives complètes
# Raison : describe() génère beaucoup de chiffres, mieux dans un fichier
donnees_reseau.describe().to_csv('data/statistiques_descriptives.csv')

print("   - data/apercu_donnees.csv")
print("   - data/statistiques_descriptives.csv")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Avoir un premier aperçu visuel des données
# CE QU'ON FAIT : Afficher 10 lignes avec seulement les colonnes clés
# POURQUOI : Comprendre rapidement le contenu sans être submergé d'informations
# ----------------------------------------------------------------------------

print("\n1.2 Aperçu des premières lignes (colonnes principales):")

# Sélectionner uniquement les colonnes les plus importantes pour la lisibilité
# Raison : Afficher 44 colonnes dans le terminal est illisible
colonnes_clés = ['dur', 'proto', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'label']
print(donnees_reseau[colonnes_clés].head(10))
print(f"\n   Note: Voir 'data/apercu_donnees.csv' pour toutes les colonnes")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Distinguer les types de variables
# CE QU'ON FAIT : Identifier variables numériques vs catégorielles
# POURQUOI : K-Means ne fonctionne qu'avec du numérique, il faut savoir
#            quelles variables seront utilisables directement
# ----------------------------------------------------------------------------

print("\n1.3 Types de variables (échantillon):")

# Extraire les colonnes numériques (int, float)
colonnes_numeriques = donnees_reseau.select_dtypes(include=[np.number]).columns

print("\n   Colonnes numériques:")
# Afficher seulement les 8 premières pour ne pas surcharger
for colonne in colonnes_numeriques[:8]:
    print(f"      {colonne:20s} - {donnees_reseau[colonne].dtype}")
print("      ...")

# Extraire les colonnes catégorielles (object = texte)
colonnes_categorielles = donnees_reseau.select_dtypes(include=['object']).columns

print("\n   Colonnes catégorielles:")
for colonne in colonnes_categorielles:
    print(f"      {colonne:20s} - {donnees_reseau[colonne].dtype}")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Comprendre la distribution statistique des données
# CE QU'ON FAIT : Afficher mean, std, min, max, quartiles
# POURQUOI : Détecter les problèmes (valeurs anormales, échelles différentes)
#            et comprendre les ordres de grandeur
# ----------------------------------------------------------------------------

print("\n1.4 Statistiques descriptives (variables principales):")

# Focus sur les 8 variables les plus importantes du sujet
# Raison : Afficher 40+ colonnes de stats serait illisible
variables_statistiques = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate', 'sttl', 'dttl']
print(donnees_reseau[variables_statistiques].describe().round(2))
print(f"\n   Note: Voir 'data/statistiques_descriptives.csv' pour toutes les variables")


variables_sujet_tp = ['dur', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 
                      'spkts', 'dpkts', 'rate']

print("\n2.1 Vérification des variables mentionnées dans le sujet du TP:")
for variable in variables_sujet_tp:
    if variable in donnees_reseau.columns:
        type_variable = donnees_reseau[variable].dtype
        print(f"   - {variable}: {type_variable}")
    else:
        print(f"   X {variable}: NON TROUVÉE")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Séparer numériques et catégorielles
# CE QU'ON FAIT : Utiliser select_dtypes() pour classifier automatiquement
# POURQUOI : Les variables catégorielles (proto, service, state) ne peuvent pas
#            être utilisées directement dans K-Means, elles nécessitent un encodage
# ----------------------------------------------------------------------------

print("\n2.2 Classification automatique des variables:")

# Extraire toutes les colonnes numériques
# Raison : K-Means calcule des distances euclidiennes, donc besoin de nombres
variables_numeriques = donnees_reseau.select_dtypes(include=[np.number]).columns.tolist()

# Extraire toutes les colonnes catégorielles (texte)
# Raison : Ces variables devront être encodées pour être utilisées
variables_categorielles = donnees_reseau.select_dtypes(include=['object']).columns.tolist()

print(f"\n   Variables numériques ({len(variables_numeriques)}):")
# Afficher seulement les 10 premières pour la lisibilité
print(f"   {', '.join(variables_numeriques[:10])}")
if len(variables_numeriques) > 10:
    print(f"   ... et {len(variables_numeriques) - 10} autres")

print(f"\n   Variables catégorielles ({len(variables_categorielles)}):")
print(f"   {', '.join(variables_categorielles)}")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Créer la liste finale des variables pour le clustering
# CE QU'ON FAIT : Exclure 'id' (identifiant) et 'label' (variable cible)
# POURQUOI : 
#   - 'id' : simple numéro d'identification, aucune valeur pour le clustering
#   - 'label' : c'est la variable qu'on cherche à prédire (0=normal, 1=attaque)
#              l'inclure serait de la triche (on ferait du supervisé, pas du clustering)
# ----------------------------------------------------------------------------

print("\n2.3 Sélection des variables numériques pour le clustering:")

# Filtrer les variables numériques en excluant id et label
# Raison : Ces colonnes ne représentent pas des caractéristiques du trafic
variables_pour_clustering = [colonne for colonne in variables_numeriques 
                             if colonne not in ['id', 'label']]

print(f"   Total: {len(variables_pour_clustering)} variables retenues")
print(f"   {', '.join(variables_pour_clustering)}")


# ============================================================================
# 3. ANALYSE DE LA DISTRIBUTION DES VARIABLES NUMÉRIQUES
# ============================================================================

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Comprendre comment les données sont distribuées
# CE QU'ON FAIT : Calculer statistiques + créer histogrammes et boxplots
# POURQUOI : 
#   - Détecter les asymétries (skew)
#   - Identifier les valeurs aberrantes
#   - Comprendre les échelles de valeurs (important pour la normalisation)
# ----------------------------------------------------------------------------

print("\n\n3. ANALYSE DE LA DISTRIBUTION DES VARIABLES NUMÉRIQUES")
print("-" * 80)

# Focus sur les 8 variables principales identifiées dans le sujet
# Raison : Analyser 40+ variables en détail serait trop long pour ce TP
variables_principales = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate', 
                         'sttl', 'dttl']

# Vérifier que ces variables existent bien dans le dataset
# Raison : Éviter les erreurs si une colonne est manquante
variables_principales = [v for v in variables_principales if v in donnees_reseau.columns]

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Obtenir les statistiques détaillées
# CE QU'ON FAIT : describe() + arrondir à 2 décimales + sauvegarder
# POURQUOI : describe() donne count, mean, std, min, 25%, 50%, 75%, max
#            Ce sont les 8 statistiques fondamentales pour comprendre une distribution
# ----------------------------------------------------------------------------

print("\n3.1 Statistiques détaillées des variables principales:")
statistiques_detaillees = donnees_reseau[variables_principales].describe().round(2)
print(statistiques_detaillees)

# Sauvegarder pour analyse approfondie ultérieure
# Raison : Pouvoir consulter ces chiffres sans relancer le script
statistiques_detaillees.to_csv('data/stats_variables_principales.csv')
print("   - Sauvegardé dans: data/stats_variables_principales.csv")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Vérifier l'intégrité des données
# CE QU'ON FAIT : Compter les valeurs manquantes (NaN)
# POURQUOI : Les valeurs manquantes posent problème pour K-Means
#            Il faudra les traiter (supprimer ou imputer) avant le clustering
# ----------------------------------------------------------------------------

print("\n3.2 Vérification de l'intégrité des données (valeurs manquantes):")

# Calculer le nombre de NaN pour chaque variable
# Raison : isnull().sum() compte les valeurs manquantes par colonne
valeurs_manquantes = donnees_reseau[variables_pour_clustering].isnull().sum()

print(f"\n   Analyse des valeurs manquantes par variable:")
# Afficher seulement les 15 premières pour la lisibilité
for colonne in variables_pour_clustering[:15]:
    nombre_nan = valeurs_manquantes[colonne]
    pourcentage_nan = (nombre_nan / nombre_connexions) * 100
    
    # N'afficher que si des valeurs manquantes sont détectées
    if nombre_nan > 0:
        print(f"   {colonne:25s}: {nombre_nan:6d} ({pourcentage_nan:5.2f}%)")

# Message de confirmation si pas de valeurs manquantes
# Raison : Rassurer l'utilisateur sur la qualité des données
if valeurs_manquantes.sum() == 0:
    print("   - Aucune valeur manquante détectée! Données complètes.")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Visualiser graphiquement les distributions
# CE QU'ON FAIT : Créer des histogrammes (distribution de fréquence)
# POURQUOI : 
#   - Histogramme montre si distribution normale, asymétrique, bimodale...
#   - Permet de voir visuellement les problèmes (outliers, échelles)
#   - Plus intuitif que les chiffres seuls
# ----------------------------------------------------------------------------

print("\n3.3 Création des graphiques de distribution...")

# Créer une grille 3x3 pour afficher 9 histogrammes
# Raison : figsize=(15,12) donne une bonne taille pour voir les détails
figure_histogrammes, axes_histogrammes = plt.subplots(3, 3, figsize=(15, 12))
figure_histogrammes.suptitle('Distribution des variables principales', fontsize=16)

# Boucler sur les 8 variables principales pour créer les histogrammes
for index, variable in enumerate(variables_principales[:9]):
    # Calculer position dans la grille 3x3
    ligne = index // 3
    colonne = index % 3
    axe = axes_histogrammes[ligne, colonne]
    
    # Créer l'histogramme
    # bins=50 : diviser les données en 50 intervalles pour un bon détail
    # edgecolor='black' : bordures noires pour mieux voir les barres
    # alpha=0.7 : légère transparence pour un rendu plus professionnel
    donnees_reseau[variable].hist(bins=50, ax=axe, edgecolor='black', alpha=0.7)
    axe.set_title(f'Distribution de {variable}')
    axe.set_xlabel(variable)
    axe.set_ylabel('Fréquence')
    axe.grid(True, alpha=0.3)  # Grille légère pour faciliter la lecture

plt.tight_layout()  # Ajuster automatiquement l'espacement
plt.savefig('data/distribution_variables_principales.png', dpi=300, bbox_inches='tight')
print("   - Graphique sauvegardé: data/distribution_variables_principales.png")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Visualiser les valeurs aberrantes (outliers)
# CE QU'ON FAIT : Créer des boxplots (boîtes à moustaches)
# POURQUOI : 
#   - Boxplot montre médiane, quartiles, et outliers
#   - Outliers = points au-delà des moustaches (Q1-1.5*IQR ou Q3+1.5*IQR)
#   - Important car outliers peuvent fausser le clustering
# ----------------------------------------------------------------------------

# Créer une nouvelle grille 3x3 pour les boxplots
figure_boxplots, axes_boxplots = plt.subplots(3, 3, figsize=(15, 12))
figure_boxplots.suptitle('Boxplots des variables principales (détection des valeurs extrêmes)', 
                         fontsize=16)

for index, variable in enumerate(variables_principales[:9]):
    ligne = index // 3
    colonne = index % 3
    axe = axes_boxplots[ligne, colonne]
    
    # Créer le boxplot
    # Raison : Le boxplot pandas affiche automatiquement Q1, médiane, Q3, outliers
    donnees_reseau.boxplot(column=variable, ax=axe)
    axe.set_title(f'Boxplot de {variable}')
    axe.set_ylabel(variable)
    axe.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/boxplots_variables_principales.png', dpi=300, bbox_inches='tight')
print("   - Graphique sauvegardé: data/boxplots_variables_principales.png")



# ============================================================================
# 4. DÉTECTION ET ANALYSE DES VALEURS ABERRANTES
# ============================================================================

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Quantifier précisément les outliers
# CE QU'ON FAIT : Méthode IQR (InterQuartile Range)
# POURQUOI : 
#   - IQR est robuste (pas influencé par les outliers eux-mêmes)
#   - Standard statistique : outlier si < Q1-1.5*IQR ou > Q3+1.5*IQR
#   - Important pour K-Means car les outliers peuvent créer des clusters artificiels
# ----------------------------------------------------------------------------

print("\n\n4. DÉTECTION ET ANALYSE DES VALEURS ABERRANTES")
print("-" * 80)

print("\n4.1 Analyse quantitative avec la méthode IQR (InterQuartile Range):")
print("   Définition: Valeur aberrante si > Q3 + 1.5×IQR ou < Q1 - 1.5×IQR")
print("   Q1 = 25e percentile, Q3 = 75e percentile, IQR = Q3 - Q1")

for variable in variables_principales:
    # Calculer les quartiles
    # Raison : quantile(0.25) = Q1 (25% des valeurs sont en dessous)
    #          quantile(0.75) = Q3 (75% des valeurs sont en dessous)
    premier_quartile = donnees_reseau[variable].quantile(0.25)
    troisieme_quartile = donnees_reseau[variable].quantile(0.75)
    
    # Calculer l'écart interquartile
    # Raison : IQR mesure la dispersion des 50% centraux des données
    ecart_interquartile = troisieme_quartile - premier_quartile
    
    # Calculer les limites selon la règle de Tukey
    # Raison : 1.5 * IQR est le coefficient standard pour détecter les outliers
    limite_inferieure = premier_quartile - 1.5 * ecart_interquartile
    limite_superieure = troisieme_quartile + 1.5 * ecart_interquartile
    
    # Identifier les valeurs aberrantes
    # Raison : Une valeur est aberrante si elle sort de ces limites
    valeurs_aberrantes = donnees_reseau[
        (donnees_reseau[variable] < limite_inferieure) | 
        (donnees_reseau[variable] > limite_superieure)
    ]
    
    nombre_outliers = len(valeurs_aberrantes)
    pourcentage_outliers = (nombre_outliers / nombre_connexions) * 100
    
    print(f"\n   {variable}:")
    print(f"      - Q1: {premier_quartile:.2f}, Q3: {troisieme_quartile:.2f}, IQR: {ecart_interquartile:.2f}")
    print(f"      - Limites acceptables: [{limite_inferieure:.2f}, {limite_superieure:.2f}]")
    print(f"      - Valeurs aberrantes: {nombre_outliers} ({pourcentage_outliers:.2f}%)")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Détecter les valeurs problématiques spécifiques
# CE QU'ON FAIT : Compter les zéros et valeurs négatives
# POURQUOI : 
#   - Zéros peuvent être normaux (pas de trafic) ou indiquer des erreurs
#   - Valeurs négatives sont souvent des erreurs de mesure
#   - Important pour décider d'un prétraitement
# ----------------------------------------------------------------------------

print("\n4.2 Analyse des valeurs nulles et négatives (possibles erreurs de mesure):")

# Rechercher valeurs nulles et négatives pour chaque variable
for variable in variables_principales:
    nombre_zeros = (donnees_reseau[variable] == 0).sum()
    nombre_negatifs = (donnees_reseau[variable] < 0).sum()
    pourcentage_zeros = (nombre_zeros / nombre_connexions) * 100
    
    # N'afficher que si des valeurs problématiques sont trouvées
    # Raison : Éviter d'encombrer l'affichage avec des lignes non pertinentes
    if nombre_zeros > 0 or nombre_negatifs > 0:
        print(f"   {variable}: {nombre_zeros} zéros ({pourcentage_zeros:.2f}%), {nombre_negatifs} négatifs")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Comprendre les relations entre variables
# CE QU'ON FAIT : Calculer la matrice de corrélation de Pearson
# POURQUOI : 
#   - Corrélation forte (proche de 1 ou -1) = redondance d'information
#   - Variables très corrélées peuvent être combinées ou une peut être supprimée
#   - Aide à comprendre quelles variables varient ensemble
# ----------------------------------------------------------------------------

print("\n4.3 Analyse des corrélations entre variables:")
print("   Création de la matrice de corrélation de Pearson...")
print("   (Corrélation de -1 = relation inversée parfaite, +1 = relation directe parfaite)")

# Créer le heatmap de corrélation
# Raison : Visualisation colorée plus intuitive qu'une matrice de chiffres
plt.figure(figsize=(14, 10))

# Calculer la matrice de corrélation
# Raison : corr() calcule le coefficient de Pearson entre chaque paire de variables
matrice_correlation = donnees_reseau[variables_principales].corr()

# Créer le heatmap avec seaborn
# annot=True : afficher les valeurs numériques dans chaque case
# fmt='.2f' : format avec 2 décimales
# cmap='coolwarm' : palette de couleurs (bleu=négatif, rouge=positif)
# center=0 : centrer la palette sur 0 (corrélations nulles)
# square=True : cases carrées pour meilleure lisibilité
sns.heatmap(matrice_correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Matrice de corrélation des variables principales')
plt.tight_layout()

# Sauvegarder le graphique
# dpi=300 : haute résolution pour impression/rapport
plt.savefig('data/matrice_correlation.png', dpi=300, bbox_inches='tight')
print("   - Graphique sauvegardé: data/matrice_correlation.png")

# Afficher aussi la matrice dans le terminal pour consultation rapide
print("\n   Matrice de corrélation (valeurs numériques):")
print(matrice_correlation.round(2))

# Sauvegarder la matrice en CSV pour analyse Excel
matrice_correlation.to_csv('data/matrice_correlation.csv')
print("   - Matrice sauvegardée: data/matrice_correlation.csv")


# ============================================================================
# 5. JUSTIFICATION DES CHOIX DE VARIABLES ET RÉPONSES AUX QUESTIONS DU TP
# ============================================================================

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Répondre aux questions du TP de manière argumentée
# CE QU'ON FAIT : Explications méthodologiques basées sur l'analyse
# POURQUOI : Justifier scientifiquement chaque décision prise
# ----------------------------------------------------------------------------

print("\n\n5. JUSTIFICATION DES VARIABLES ET RÉPONSES AUX QUESTIONS DU TP")
print("-" * 80)

print("\n5.1 Variables pertinentes retenues pour la segmentation du trafic réseau:")
print("""
   DÉCISION : Nous retenons les 8 variables numériques suivantes
   
   - dur (durée)           : Temps de la connexion en secondes
                             → Distingue connexions courtes (scans) vs longues (téléchargements)
   
   - sbytes, dbytes        : Volume de données source/destination en octets
                             → Identifie trafic léger vs volumineux
   
   - spkts, dpkts          : Nombre de paquets source/destination
                             → Caractérise la fragmentation du trafic
   
   - rate                  : Taux de transfert (paquets/seconde)
                             → Mesure l'intensité du flux réseau
   
   - sttl, dttl            : Time-To-Live source/destination
                             → Indicateur de distance réseau et routage
   
   JUSTIFICATION MÉTIER :
   Ces variables permettent d'identifier différents profils de trafic:
   - Trafic léger (navigation) vs volumineux (streaming, téléchargement)
   - Connexions courtes (requêtes HTTP) vs longues (sessions persistantes)
   - Trafic asymétrique (client/serveur) vs symétrique (P2P)
   - Comportements anormaux (scans de ports, attaques volumétriques)
""")

print("\n5.2 Variables à exclure de l'analyse K-Means:")
print("""
   EXCLUSIONS OBLIGATOIRES :
   
   X id                    : Simple identifiant technique sans signification réseau
                             → Aucune valeur prédictive, pure numérotation
   
   X label                 : Variable cible (0=normal, 1=attaque)
                             → L'inclure transformerait le clustering non supervisé
                               en apprentissage supervisé (triche!)
   
   X attack_cat            : Catégorie d'attaque (quand label=1)
                             → Information à prédire, pas à utiliser en entrée
   
   VARIABLES CATÉGORIELLES (nécessitent encodage) :
   
   X proto                 : Protocole réseau (tcp, udp, icmp...)
   X service               : Service applicatif (http, ftp, ssh...)
   X state                 : État de la connexion (FIN, INT, CON...)"
   
   → Ces variables contiennent des informations utiles mais K-Means
     ne peut traiter que des nombres. Elles peuvent être ajoutées
     après encodage (One-Hot ou Label Encoding) dans une analyse ultérieure.
""")

print("\n5.3 RÉPONSE À LA QUESTION : Pourquoi normaliser certaines variables?")
print("""
   RÉPONSE COMPLÈTE :
   
   La normalisation est ABSOLUMENT ESSENTIELLE pour K-Means car :
   
   1. PROBLÈME MATHÉMATIQUE : K-Means utilise la distance euclidienne
      Distance = √[(x₁-x₂)² + (y₁-y₂)² + ...]
      
   2. PROBLÈME D'ÉCHELLE : Nos variables ont des échelles très différentes
      - dur  : de 0 à quelques secondes (ordre de grandeur : 10⁰)
      - sbytes/dbytes : de 0 à plusieurs millions (ordre de grandeur : 10⁶)
      - rate : de 0 à plusieurs milliers (ordre de grandeur : 10³)
   
   3. CONSÉQUENCE SANS NORMALISATION :
      Les variables à grande échelle (sbytes, dbytes) DOMINENT le calcul
      de distance et rendent les autres variables (dur, rate) INVISIBLES
      
      Exemple concret :
      Distance entre 2 connexions serait presque uniquement basée sur
      la différence de bytes, ignorant totalement la durée!
   
   4. SOLUTION : Standardisation (z-score)
      Formule : z = (x - μ) / σ
      Où μ = moyenne, σ = écart-type
      
      → Transforme chaque variable pour avoir :
        - Moyenne = 0
        - Écart-type = 1
      
      → Toutes les variables ont maintenant le MÊME POIDS dans le clustering
   
   5. MOMENT D'APPLICATION :
      La normalisation sera appliquée dans la Partie 2 (Préparation des données)
      après cette analyse exploratoire.
""")

print("\n5.4 RÉPONSE À LA QUESTION : Pourquoi encoder les variables catégorielles?")
print("""
   RÉPONSE COMPLÈTE :
   
   Les variables catégorielles (proto, service, state) doivent être
   encodées pour être utilisables dans K-Means car :
   
   1. LIMITATION ALGORITHMIQUE :
      K-Means ne fonctionne qu'avec des NOMBRES (calcul de moyennes)
      Les chaînes de caractères ('tcp', 'http', 'FIN') ne peuvent pas être
      moyennées ou utilisées dans des calculs de distance
   
   2. INFORMATION PERDUE SANS ENCODAGE :
      Ces variables contiennent des informations PERTINENTES :
      - proto : type de protocole (TCP, UDP, ICMP...)
        → Chaque protocole a un comportement différent
      - service : service réseau (HTTP, FTP, DNS, SSH...)
        → Indicateur du type d'application
      - state : état de la connexion (FIN=terminée, CON=établie...)
        → Révèle des anomalies (nombreuses connexions non terminées)
   
   3. MÉTHODES D'ENCODAGE POSSIBLES :
   
      A) ONE-HOT ENCODING (recommandé pour K-Means)
         Principe : Créer une colonne binaire (0/1) par catégorie
         Exemple pour 'proto':
            ┌──────┬─────────┬─────────┬──────────┐
            │proto │proto_tcp│proto_udp│proto_icmp│
            ├──────┼─────────┼─────────┼──────────┤
            │tcp   │    1    │    0    │    0     │
            │udp   │    0    │    1    │    0     │
            │icmp  │    0    │    0    │    1     │
            └──────┴─────────┴─────────┴──────────┘
         Avantage : Pas d'ordre arbitraire entre catégories
         Inconvénient : Augmente la dimensionnalité
      
      B) LABEL ENCODING (plus simple mais limite)
         Principe : Assigner un entier à chaque catégorie
         Exemple : tcp=0, udp=1, icmp=2
         Avantage : Une seule colonne
         Inconvénient : Crée un ordre artificiel (tcp < udp < icmp)
   
   4. STRATÉGIE POUR CE TP :
      - Partie 1 (actuelle) : Analyser uniquement variables numériques
      - Partie 2 : Appliquer K-Means sur variables numériques normalisées
      - Extension possible : Ajouter variables catégorielles encodées
        et comparer les résultats
""")

# ----------------------------------------------------------------------------
# CE QU'ON VEUT : Résumer les résultats de l'analyse exploratoire
# CE QU'ON FAIT : Afficher un récapitulatif concis
# POURQUOI : Permet de vérifier rapidement que tout s'est bien passé
# ----------------------------------------------------------------------------

print("\n\n" + "=" * 80)
print("RÉSUMÉ DE L'ANALYSE EXPLORATOIRE - PARTIE 1 COMPLÉTÉE")
print("=" * 80)

print(f"\nDONNÉES ANALYSÉES:")
print(f"   - Dataset chargé: {nombre_connexions:,} connexions réseau")
print(f"   - Variables totales: {nombre_variables}")
print(f"   - Variables numériques: {len(variables_numeriques)}")
print(f"   - Variables catégorielles: {len(variables_categorielles)}")

print(f"\nVARIABLES POUR LE CLUSTERING:")
print(f"   - Variables retenues: {len(variables_pour_clustering)}")
print(f"   - Variables principales analysées: {len(variables_principales)}")
print(f"   - Valeurs manquantes: {donnees_reseau[variables_pour_clustering].isnull().sum().sum()}")

print(f"\nFICHIERS GÉNÉRÉS (dossier data/):")
print(f"   - apercu_donnees.csv - Échantillon des données")
print(f"   - statistiques_descriptives.csv - Stats complètes")
print(f"   - stats_variables_principales.csv - Stats variables clés")
print(f"   - distribution_variables_principales.png - Histogrammes")
print(f"   - boxplots_variables_principales.png - Détection outliers")
print(f"   - matrice_correlation.png - Heatmap corrélations")
print(f"   - matrice_correlation.csv - Matrice numérique")

print(f"\nQUESTIONS DU TP RÉPONDUES:")
print(f"   - Pourquoi normaliser certaines variables?")
print(f"   - Pourquoi encoder les variables catégorielles?")

print(f"\nPROCHAINE ÉTAPE:")
print(f"   -> Partie 2 - Préparation des données")
print(f"     (Normalisation StandardScaler et sélection finale des variables)")
print(f"\n" + "=" * 80)
