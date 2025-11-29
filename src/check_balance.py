import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration des chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')

def verifier_equilibre():
    print("--- ⚖️ VÉRIFICATION DE L'ÉQUILIBRE DES CLASSES ---")
    
    # 1. Chargement
    if not os.path.exists(DATA_PATH):
        print("❌ Fichier introuvable. Avez-vous lancé preprocessing.py ?")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    # 2. Calcul des statistiques
    # On compte combien de fois chaque chiffre (0 à 9) apparaît
    counts = df['Label'].value_counts().sort_index()
    percentages = df['Label'].value_counts(normalize=True).sort_index() * 100
    
    print("\n📋 Répartition numérique :")
    print(f"{'Classe':<10} | {'Nombre (Lignes)':<15} | {'Pourcentage':<15}")
    print("-" * 45)
    
    for label, count in counts.items():
        pct = percentages[label]
        # On essaie de deviner le nom (basé sur votre output précédent)
        nom = "Inconnu"
        if label == 0: nom = "Normal (BENIGN)"
        elif label == 2: nom = "DDoS" # Exemple basé sur la majorité
        
        print(f"Classe {label:<3} | {count:<15} | {pct:.2f}%")

    # 3. Visualisation Graphique
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Label', data=df, palette='viridis')
    plt.title('Distribution des Classes (Équilibre du Dataset)')
    plt.xlabel('Classes (0=Normal, 1-9=Attaques)')
    plt.ylabel('Nombre de connexions')
    plt.yscale('log') # ASTUCE : Échelle logarithmique pour voir les petites classes
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    print("\n📈 Une fenêtre graphique va s'ouvrir...")
    plt.show()

if __name__ == "__main__":
    verifier_equilibre()