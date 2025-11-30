import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration des chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DATA_PATH = os.path.join(project_root, 'data', 'processed', 'dataset_cleaned.csv')

def audit_class_distribution():
    """ 
    Analyse la distribution des classes (Labels) dans le dataset.
    Génère des statistiques textuelles et un graphique logarithmique 
    pour visualiser le déséquilibre entre trafic normal et attaques rares.
    """
    print("[INFO] Démarrage de l'audit de distribution des données...")
    
    # 1. Chargement des données
    if not os.path.exists(DATA_PATH):
        print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    # 2. Calcul des statistiques descriptives
    # Tri par index pour avoir les classes dans l'ordre (0, 1, 2...)
    counts = df['Label'].value_counts().sort_index()
    percentages = df['Label'].value_counts(normalize=True).sort_index() * 100
    
    print("\n[RESULTATS] Répartition des classes :")
    print(f"{'Classe ID':<10} | {'Nombre (Occurrences)':<20} | {'Pourcentage':<15}")
    print("-" * 50)
    
    for label, count in counts.items():
        pct = percentages[label]
        print(f"Classe {label:<3} | {count:<20} | {pct:.2f}%")

    # 3. Visualisation Graphique
    print("\n[INFO] Génération du graphique de distribution...")
    plt.figure(figsize=(12, 6))
    
    # Utilisation de hue=x pour éviter le FutureWarning de Seaborn
    sns.countplot(x='Label', data=df, hue='Label', legend=False, palette='viridis')
    
    plt.title('Distribution des Classes (Échelle Logarithmique)')
    plt.xlabel('Identifiant de la Classe (0=Normal, 1-9=Attaques)')
    plt.ylabel('Nombre de connexions (Log Scale)')
    
    # Application d'une échelle logarithmique pour visualiser les classes minoritaires
    # Sans cela, les attaques rares (Heartbleed, etc.) seraient invisibles
    plt.yscale('log') 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    print("[INFO] Affichage du graphique.")
    plt.show()

if __name__ == "__main__":
    audit_class_distribution()