import pandas as pd
import numpy as np
import os
import joblib  # Librairie pour sauvegarder le modèle (sérialisation)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

DATA_PATH = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')
# Le fichier .pkl est le "cerveau" que nous utiliserons dans l'application finale
MODEL_PATH = os.path.join(project_root, 'models', 'random_forest_final.pkl')
IMG_PATH = os.path.join(project_root, 'docs', 'images')

def train_final_model():
    """ 
    Entraîne le modèle Random Forest définitif avec les hyperparamètres optimisés.
    Génère le fichier .pkl pour la mise en production (Phase 5).
    """
    print("[INFO] Initialisation de l'entraînement du modèle final...")

    # 1. Chargement des données
    if not os.path.exists(DATA_PATH):
        print(f"[ERREUR] Fichier de données introuvable : {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    print(f"[INFO] Données chargées. Dimensions : {X.shape}")

    # 2. Division Train/Test
    # Utilisation de stratify=y pour maintenir la proportion des attaques rares
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Configuration du modèle
    # Nous utilisons ici les résultats de l'optimisation (GridSearch - Phase 3.4)
    print("[INFO] Configuration du modèle avec hyperparamètres optimisés...")
    
    model = RandomForestClassifier(
        n_estimators=100,           # Paramètre optimal trouvé
        max_depth=20,               # Paramètre optimal pour éviter l'overfitting
        criterion='gini',           # Critère de pureté
        class_weight='balanced',    # Gestion du déséquilibre des classes (Crucial pour Heartbleed/Infiltration)
        n_jobs=-1,                  # Utilisation de tous les cœurs du CPU
        random_state=42             # Reproductibilité
    )
    
    # 4. Entraînement
    print("[INFO] Démarrage de l'apprentissage sur l'ensemble d'entraînement...")
    model.fit(X_train, y_train)
    print("[INFO] Entraînement terminé.")

    # 5. Validation finale
    print("[INFO] Évaluation des performances sur le jeu de test...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULTAT] Accuracy finale : {acc:.4f} ({(acc*100):.2f}%)")

    # Génération de la matrice de confusion pour le rapport
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion - Modèle Final Optimisé')
    
    os.makedirs(IMG_PATH, exist_ok=True)
    save_img = os.path.join(IMG_PATH, 'confusion_matrix_final.png')
    plt.savefig(save_img)
    print(f"[INFO] Matrice de confusion exportée : {save_img}")

    # 6. Sauvegarde du modèle (Mise en production)
    print("[INFO] Exportation du modèle...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    print(f"[SUCCES] Modèle sauvegardé avec succès : {MODEL_PATH}")
    print("[INFO] Prêt pour l'intégration dans l'IDS (Phase 5).")

if __name__ == "__main__":
    train_final_model()