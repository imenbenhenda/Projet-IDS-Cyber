import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Configuration des chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')

def check_generalization():
    """ 
    Vérifie la capacité de généralisation du modèle.
    Compare les performances sur le jeu d'entraînement (Train) vs Test.
    Un écart faible (< 5%) indique l'absence de sur-apprentissage (Overfitting).
    """
    print("[INFO] Démarrage de l'analyse de généralisation (Overfitting check)...")
    
    # 1. Chargement partiel pour audit rapide
    if not os.path.exists(DATA_PATH):
        print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
        return

    # Echantillonnage à 10% suffisant pour vérifier la convergence
    df = pd.read_csv(DATA_PATH).sample(frac=0.1, random_state=42)
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 2. Division Train/Test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Instanciation du modèle avec régularisation
    # Utilisation de max_depth=20 (paramètre optimisé) pour limiter la complexité de l'arbre
    print("[INFO] Entraînement du modèle de test...")
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    # 4. Calcul des scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    gap = train_score - test_score
    
    # 5. Affichage du rapport
    print("\n[RESULTATS] Métriques de performance :")
    print(f" - Score Train (Apprentissage) : {train_score:.4f} ({(train_score*100):.2f}%)")
    print(f" - Score Test (Généralisation) : {test_score:.4f} ({(test_score*100):.2f}%)")
    print(f" - Ecart (Gap)                 : {gap:.4f} ({(gap*100):.2f}%)")
    
    # Seuil de tolérance industriel : 5%
    if gap > 0.05:
        print("\n[ATTENTION] Risque de sur-apprentissage (Overfitting) détecté.")
        print(" -> Recommandation : Réduire 'max_depth' ou augmenter le volume de données.")
    else:
        print("\n[SUCCES] Le modèle généralise correctement.")
        print(" -> L'écart entre Train et Test est négligeable (< 5%).")

if __name__ == "__main__":
    check_generalization()