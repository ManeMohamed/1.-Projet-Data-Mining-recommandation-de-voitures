# Vehicle Recommendation System — Data Mining Project

Projet réalisé dans le cadre du Master **Sciences de Données pour une Industrie Intelligente (ENSAM Meknès)**.  
Objectif : recommander les voitures les plus adaptées aux clients en fonction de leurs caractéristiques et du catalogue produit.

---

## 🎯 Objectifs du projet
- Analyser et nettoyer les données clients et le catalogue véhicules  
- Explorer les données (EDA) avec visualisations avancées  
- Réduire la dimension avec **t-SNE**  
- Déterminer le meilleur nombre de clusters avec la **silhouette score**  
- Appliquer le clustering uniquement sur les **voitures neuves**  
- Entraîner un modèle supervisé (Random Forest ou Logistic Regression)  
- Recommander un type de voiture pour un nouveau client  
- Construire une petite application Python / Flask pour tester la recommandation  

---

## 📊 Jeu de données
- **dataset_clients.csv** : informations sur les clients  
- **dataset_catalogue.csv** : liste des véhicules  
- Variables utilisées :
  - Sexe, catégorie socio-pro, situation familiale  
  - Prix, puissance, marque, carburant, type, gamme…

---

## 🧹 1. Preprocessing
- Encodage des variables catégorielles  
- Standardisation / normalisation  
- Filtration : sélection uniquement des voitures **neuves** pour le clustering  
- Gestion des missing values  

Fichier : `src/preprocessing.py`

---

## 🔍 2. Analyse Exploratoire (EDA)
Visualisations :
- Histogrammes / distribution des prix  
- Boxplots par carburant / marque  
- Heatmap des corrélations  
- Countplots des catégories clients  
- Scatter plots (prix vs puissance)

Notebook : `notebooks/02_eda_visualizations.ipynb`

---

## 🌀 3. Réduction de dimension (t-SNE)
- t-SNE pour visualiser les véhicules en 2D  
- Paramètres optimisés : perplexity, learning rate  
- Sauvegarde du graphique

Output : `results/tsne_plot.png`

Fichier : `src/tsne.py`

---

## 📈 4. Clustering des voitures neuves
- Méthode : **K-Means**  
- Sélection du meilleur k via **silhouette score**  
- Comparaison de k = 2 à 10  
- Visualisation finale en 2D via t-SNE

Output : `results/silhouette_scores.png`

Notebook : `notebooks/04_clustering_new_cars.ipynb`

---

## 🤖 5. Modèle supervisé (Random Forest)
Objectif : prédire la catégorie de voiture idéale pour un client.

- Pipeline scikit-learn (encoder + scaler + RandomForestClassifier)  
- Evaluation : accuracy, confusion matrix, classification report  
- Importance des features

Output : `results/model_metrics.png`

Notebook : `notebooks/05_model_prediction.ipynb`

---

## 🚀 6. Mini application Flask
Permet d’entrer :
- sexe  
- âge  
- revenu  
- situation familiale  
- préférences  

Et retourne :  
➡️ **la catégorie de voiture recommandée**

Code : `app/app.py`

---

## 📁 Structure du repository

