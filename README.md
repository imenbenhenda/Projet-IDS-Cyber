# 🛡️ Projet IDS : Détection d’Intrusions Intelligente

> **Projet Académique de Cybersécurité**

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
├── data/                 
│   ├── raw/               
│   └── processed/         
│
├── docs/                  
│   └── rapport_phase1/  
│
├── notebooks/             
│   ├── 01_exploration.ipynb
│   └── 02_nettoyage.ipynb
│
├── src/                  
│   ├── preprocessing.py   
│   └── train.py           
│
├── models/              
└── requirements.txt       