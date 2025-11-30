import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# Configuration des chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')
IMG_PATH = os.path.join(project_root, 'docs', 'images')

def compare_models():
    """
    Compare les performances de trois approches techniques :
    1. Supervisé (Random Forest)
    2. Deep Learning (MLP)
    3. Non Supervisé (Isolation Forest)
    """
    print("[INFO] Démarrage du comparatif des modèles...")

    # 1. Chargement et Echantillonnage
    if not os.path.exists(DATA_PATH):
        print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Echantillonnage à 10% pour réduire le temps de calcul lors de la phase comparative
    df_sample = df.sample(frac=0.1, random_state=42)
    print(f"[INFO] Données chargées (Echantillon 10%) : {df_sample.shape}")

    X = df_sample.drop('Label', axis=1)
    y = df_sample['Label']

    # Division Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Création d'une cible binaire pour l'évaluation commune (0=Normal, 1=Attaque)
    # Nécessaire car l'Isolation Forest ne prédit pas les classes multi-labels
    y_test_binary = y_test.apply(lambda x: 0 if x == 0 else 1)

    results = []

    # -------------------------------------------------------
    # 1. Random Forest (Approche Supervisée)
    # -------------------------------------------------------
    print("\n[1/3] Entraînement Random Forest (Supervisé)...")
    start = time.time()
    
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    end = time.time()
    
    # Binarisation des prédictions pour comparaison homogène
    y_pred_binary = [0 if x == 0 else 1 for x in y_pred]
    
    acc = accuracy_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    results.append({'Modèle': 'Supervisé (Random Forest)', 'Accuracy': acc, 'F1-Score': f1, 'Temps': end-start})
    print(f" - Accuracy : {acc:.4f} | Temps : {end-start:.2f}s")

    # -------------------------------------------------------
    # 2. MLP Classifier (Deep Learning)
    # -------------------------------------------------------
    print("\n[2/3] Entraînement MLP (Deep Learning)...")
    start = time.time()
    
    # Configuration avec early_stopping pour optimiser le temps de convergence
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=300, early_stopping=True, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    end = time.time()
    
    y_pred_binary = [0 if x == 0 else 1 for x in y_pred]
    
    acc = accuracy_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    results.append({'Modèle': 'Deep Learning (MLP)', 'Accuracy': acc, 'F1-Score': f1, 'Temps': end-start})
    print(f" - Accuracy : {acc:.4f} | Temps : {end-start:.2f}s")

    # -------------------------------------------------------
    # 3. Isolation Forest (Approche Non Supervisée)
    # -------------------------------------------------------
    print("\n[3/3] Entraînement Isolation Forest (Non Supervisé)...")
    start = time.time()
    
    # Entraînement uniquement sur X_train (sans labels)
    iso = IsolationForest(n_estimators=50, contamination=0.4, n_jobs=-1, random_state=42)
    iso.fit(X_train)
    y_pred_iso = iso.predict(X_test)
    
    end = time.time()

    # Mapping des prédictions : 1 (Normal) -> 0, -1 (Anomalie) -> 1
    y_pred_binary = [0 if x == 1 else 1 for x in y_pred_iso]

    acc = accuracy_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    results.append({'Modèle': 'Non-Sup (Isolation Forest)', 'Accuracy': acc, 'F1-Score': f1, 'Temps': end-start})
    print(f" - Accuracy : {acc:.4f} | Temps : {end-start:.2f}s")

    # -------------------------------------------------------
    # Synthèse des résultats
    # -------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n[INFO] Tableau comparatif final :")
    print(results_df)

    # Génération du graphique
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Modèle', y='Accuracy', data=results_df, palette='viridis')
    plt.title('Comparaison des Performances : Supervisé vs DL vs Non-Supervisé')
    plt.ylim(0.5, 1.0)
    
    os.makedirs(IMG_PATH, exist_ok=True)
    save_path = os.path.join(IMG_PATH, 'comparaison_3_types.png')
    plt.savefig(save_path)
    print(f"\n[INFO] Graphique sauvegardé : {save_path}")

if __name__ == "__main__":
    compare_models()