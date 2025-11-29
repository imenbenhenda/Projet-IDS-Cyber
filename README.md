# 🛡️ Projet IDS : Détection d’Intrusions Intelligente

> **Projet Académique de Cybersécurité** > **Master / Année de spécialisation**

Ce projet vise à concevoir et développer un **Système de Détection d'Intrusions (IDS)** de nouvelle génération. Contrairement aux systèmes classiques basés sur des signatures (comme Snort), notre solution utilise le **Machine Learning** et le **Deep Learning** pour apprendre le comportement normal du réseau et identifier les anomalies (attaques inconnues ou zero-day).

---

## 🎯 Objectifs

L'objectif principal est de classifier le trafic réseau en **Trafic Normal** ou **Attaque** (DDoS, Botnet, PortScan, Brute Force) à partir de flux réseaux réels.

* **Collecter & Préparer :** Nettoyer et normaliser le dataset **CICIDS2017**.
* **Modéliser :** Entraîner des modèles d'IA (Random Forest, SVM, Réseaux de Neurones).
* **Évaluer :** Comparer les performances (Précision, Recall, F1-Score).
* **Visualiser :** Créer une interface de détection pour un usage SOC.

---

## 📂 Structure du Projet

```text
Projet-IDS-Cybersecurite/
│
├── data/                  # Données du projet (Ignoré par Git)
│   ├── raw/               # Placer ici les fichiers CSV originaux (CICIDS2017)
│   └── processed/         # Fichiers nettoyés prêts pour l'entraînement
│
├── docs/                  # Documentation et Rapports
│   └── rapport_phase1/    # Fichiers LaTeX
│
├── notebooks/             # Jupyter Notebooks pour l'expérimentation
│   ├── 01_exploration.ipynb
│   └── 02_nettoyage.ipynb
│
├── src/                   # Code source Python modulaire
│   ├── preprocessing.py   # Scripts de nettoyage
│   └── train.py           # Scripts d'entraînement
│
├── models/                # Modèles entraîner (.pkl, .h5)
└── requirements.txt       # Liste des dépendances Python