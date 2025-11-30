import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Configuration des chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

RAW_FILE = os.path.join(project_root, 'data', 'raw', 'combine.csv')
PROCESSED_FILE = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')

def load_data():
    """ Chargement du dataset brut """
    print(f"[INFO] Chargement du fichier : {RAW_FILE}")
    
    if not os.path.exists(RAW_FILE):
        print("[ERREUR] Fichier introuvable.")
        return None
    
    df = pd.read_csv(RAW_FILE)
    print(f"[INFO] Fichier chargé. Dimensions initiales : {df.shape}")
    return df

def clean_data(df):
    """ Nettoyage des données : gestion des infinis, nulls et doublons """
    print("\n[INFO] Démarrage du nettoyage...")
    
    # Correction des noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Gestion des valeurs infinies et manquantes
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    n_nan = df.isna().sum().sum()
    df.dropna(inplace=True)
    print(f" - Valeurs NaN supprimées : {n_nan}")
    
    # Suppression des doublons
    n_dupli = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f" - Doublons supprimés : {n_dupli}")
    
    print(f"[INFO] Nettoyage terminé. Nouvelles dimensions : {df.shape}")
    return df

def transform_data(df):
    """ Encodage des labels et normalisation des features """
    print("\n[INFO] Transformation et Encodage...")
    
    # Encodage de la variable cible
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    # Mapping des classes
    print(" - Correspondance des classes :")
    for i, label in enumerate(le.classes_):
        print(f"   {label} : {i}")

    # Normalisation (MinMax Scaling)
    print(" - Normalisation des features...")
    
    X = df.drop(columns=['Label', 'Label_Encoded'])
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reconstruction du DataFrame
    df_clean = pd.DataFrame(X_scaled, columns=X.columns)
    df_clean['Label'] = df['Label_Encoded'].values
    
    return df_clean

if __name__ == "__main__":
    # Pipeline d'exécution
    df = load_data()
    
    if df is not None:
        df = clean_data(df)
        df_final = transform_data(df)
        
        # Exportation
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        
        print(f"\n[INFO] Sauvegarde du dataset traité : {PROCESSED_FILE}")
        df_final.to_csv(PROCESSED_FILE, index=False)
        print("[INFO] Traitement terminé avec succès.")