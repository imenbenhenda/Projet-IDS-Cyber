import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==========================================
# 1. CONFIGURATION DES CHEMINS 
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(script_dir)              

# Fichiers d'entrée et de sortie
RAW_FILE = os.path.join(project_root, 'data', 'raw', 'combine.csv')
PROCESSED_FILE = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')

def load_data():
    """ Étape 1 : Chargement """
    print(f"🔄 Chargement du fichier : {RAW_FILE}...")
    if not os.path.exists(RAW_FILE):
        print("❌ ERREUR : Fichier introuvable.")
        return None
    
    # On charge tout le fichier 
    df = pd.read_csv(RAW_FILE)
    print(f"✅ Fichier chargé. Taille initiale : {df.shape}")
    return df

def clean_data(df):
    """ Étape 2 : Nettoyage (Suppression valeurs aberrantes, duplication) """
    print("\n🧹 --- Nettoyage des données ---")
    
    # 1. Nettoyer les noms de colonnes (supprimer les espaces cachés)
    df.columns = df.columns.str.strip()
    
    # 2. Remplacer les valeurs infinies par NaN 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 3. Supprimer les lignes vides (NaN)
    n_nan = df.isna().sum().sum()
    df.dropna(inplace=True)
    print(f"   - Valeurs vides (NaN) supprimées : {n_nan}")
    
    # 4. Supprimer les doublons (Lignes identiques)
    n_dupli = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"   - Doublons supprimés : {n_dupli}")
    
    print(f"✅ Nettoyage terminé. Nouvelle taille : {df.shape}")
    return df

def transform_data(df):
    """ Étape 4 & 5 : Normalisation et Annotation """
    print("\n⚙️ --- Transformation et Encodage ---")
    
    # --- A. Encodage du Label (Étape 5) ---
    # Transformer "BENIGN" en 0 et "DDoS" en 1
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    # On affiche la correspondance pour comprendre
    classes = le.classes_
    print("   📝 Traduction des attaques (Encodage) :")
    for i, label in enumerate(classes):
        print(f"      - {label} est devenu le chiffre {i}")

    # --- B. Normalisation (Étape 4) ---
    # Mettre tous les chiffres entre 0 et 1 
    print("   ⚖️  Normalisation des valeurs (MinMax Scaling)...")
    
    # On sépare les caractéristiques (X) de la cible (Y)
    # On enlève 'Label' (texte) et 'Label_Encoded' (notre cible finale)
    X = df.drop(columns=['Label', 'Label_Encoded'])
    
    # On normalise X
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # On recrée un DataFrame propre
    df_clean = pd.DataFrame(X_scaled, columns=X.columns)
    
    # On rajoute la colonne cible à la fin (celle avec les chiffres 0, 1...)
    df_clean['Label'] = df['Label_Encoded'].values
    
    print("✅ Transformation terminée.")
    return df_clean

# ==========================================
# EXÉCUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":
    # 1. Charger
    df = load_data()
    
    if df is not None:
        # 2. Nettoyer
        df = clean_data(df)
        
        # 3. Transformer
        df_final = transform_data(df)
        
        # 4. Sauvegarder
        # On s'assure que le dossier 'processed' existe
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        
        print(f"\n💾 Sauvegarde du fichier propre vers : {PROCESSED_FILE}")
        df_final.to_csv(PROCESSED_FILE, index=False)
        print("🎉 SUCCÈS : La Phase 2 est terminée ! Les données sont prêtes pour l'IA.")