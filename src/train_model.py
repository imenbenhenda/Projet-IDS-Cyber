import pandas as pd
import numpy as np
import os
import joblib  # Pour sauvegarder le modèle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Détection automatique des chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

DATA_PATH = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')
MODEL_PATH = os.path.join(project_root, 'models', 'random_forest_model.pkl')
IMG_PATH = os.path.join(project_root, 'docs', 'images')

def train_and_evaluate():
    print("--- 🚀 DÉMARRAGE DE L'ENTRAÎNEMENT (PHASE 3) ---")

    # --- A. Chargement ---
    print(f"1. Chargement des données depuis {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("❌ ERREUR : Fichier introuvable.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Séparation X (Features) et y (Cible)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    print(f"   Données chargées : {X.shape}")

    # --- B. Division Train / Test ---
    print("\n2. Division du dataset (80% Train - 20% Test)...")
    # On garde 20% pour l'examen final
    # stratify=y est CRUCIAL ici : il assure qu'il y a un peu de Classe 7 dans le Train ET dans le Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   - Entraînement sur : {X_train.shape[0]} lignes")
    print(f"   - Test sur : {X_test.shape[0]} lignes")

    # --- C. Entraînement (Phase 3) ---
    print("\n3. Entraînement du modèle (Random Forest)...")
    print("   ⏳ Cela peut prendre 2 à 5 minutes selon votre PC. Patience...")
    
    # ASTUCE : class_weight='balanced' aide à détecter les attaques rares (Heartbleed, etc.)
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès !")

    # --- D. Évaluation (Phase 4) ---
    print("\n4. Évaluation des performances...")
    y_pred = model.predict(X_test)

    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"   🏆 Accuracy (Précision globale) : {acc:.4f} (soit {acc*100:.2f}%)")

    # 2. Rapport détaillé
    print("\n   📊 Rapport de classification par classe :")
    # zero_division=0 empêche le code de planter si une classe a 0 réussite
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # 3. Matrice de confusion
    print("   🖼️ Génération de la matrice de confusion...")
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion - Random Forest')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    
    # Sauvegarde de l'image
    os.makedirs(IMG_PATH, exist_ok=True)
    save_img = os.path.join(IMG_PATH, 'confusion_matrix_rf.png')
    plt.savefig(save_img)
    print(f"   Image sauvegardée dans : {save_img}")

    # --- E. Sauvegarde du Modèle ---
    print("\n5. Sauvegarde du modèle...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans : {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()