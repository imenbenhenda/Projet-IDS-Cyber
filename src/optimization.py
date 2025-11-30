import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Configuration des chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')
IMG_PATH = os.path.join(project_root, 'docs', 'images')

def optimiser_modele():
    """ Exécute l'optimisation des hyperparamètres (GridSearch) et génère la courbe ROC """
    print("[INFO] Démarrage de l'optimisation (Grid Search)...")
    
    # 1. Chargement et Echantillonnage
    if not os.path.exists(DATA_PATH): 
        print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    # Echantillonnage à 5% pour réduire le temps de calcul du GridSearch
    df_sample = df.sample(frac=0.05, random_state=42)
    
    # Filtrage des classes rares (< 2 instances) pour permettre la stratification
    class_counts = df_sample['Label'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_sample = df_sample[df_sample['Label'].isin(valid_classes)]
    
    print(f"[INFO] Echantillon filtré : {df_sample.shape[0]} lignes")
    
    X = df_sample.drop('Label', axis=1)
    y = df_sample['Label']
    
    # Division Train/Test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Configuration du GridSearch
    # Définition de l'espace de recherche des hyperparamètres
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'criterion': ['gini']
    }
    
    print(f"[INFO] Recherche des meilleurs hyperparamètres pour Random Forest...")
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Exécution du GridSearch avec validation croisée (k=3)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("[INFO] Optimisation terminée.")
    print(f" - Meilleurs paramètres : {grid_search.best_params_}")
    print(f" - Meilleur score (Accuracy) : {grid_search.best_score_:.4f}")

    # 3. Génération de la courbe ROC (Phase 4.1)
    print("[INFO] Génération de la courbe ROC...")
    
    best_model = grid_search.best_estimator_
    
    # Binarisation des labels pour le calcul ROC (0=Normal, 1=Attaque)
    y_test_bin = y_test.apply(lambda x: 0 if x == 0 else 1)
    
    # Récupération des probabilités prédites
    y_probs = best_model.predict_proba(X_test)
    
    # Probabilité d'appartenance à une classe "Attaque" (1 - probabilité classe 0)
    y_probs_attack = 1 - y_probs[:, 0]
    
    # Calcul des métriques ROC
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_probs_attack)
    roc_auc = auc(fpr, tpr)
    
    # Tracé du graphique
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbe ROC - Performance du Modèle Optimisé')
    plt.legend(loc="lower right")
    
    # Exportation de l'image
    os.makedirs(IMG_PATH, exist_ok=True)
    save_path = os.path.join(IMG_PATH, 'roc_curve.png')
    plt.savefig(save_path)
    print(f"[INFO] Courbe ROC sauvegardée : {save_path}")

if __name__ == "__main__":
    optimiser_modele()