# 🚗 Prédiction du Prix de Vente d'une Voiture d'Occasion

Projet réalisé dans le cadre du **Module 5 : Machine Learning — EHTP, MSDE Édition 7**.

## 🔗 Liens

- **Application Streamlit déployée :** https://project-ml-used-car-price-prediction.streamlit.app/
- **Dépôt GitHub :** https://github.com/badrnntest-creator/Project-ML-_-Used-Car-Price-Prediction

## 📸 Captures d'écran

### Application Streamlit

![Application Streamlit](screenshot_streamlit.png)

### Dépôt GitHub

![Dépôt GitHub](screenshot_github.png)

## 📁 Structure du projet

```
used-car-price-prediction/
├── app/               # Application Streamlit (app.py)
├── data/              # Données (X_train, X_test, y_train, y_test)
├── models/            # Pipeline et modèles sérialisés (pkl)
├── notebooks/         # Notebooks (EDA, Preprocessing, Modeling, Tuning)
├── report/            # Rapport et graphes du projet
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Exécution

```bash
streamlit run app/app.py
```

## 🏆 Résultat

| Modèle | R² | RMSE |
|--------|----|------|
| XGBoost (après tuning) | 0.8679 | 0.4279 |

## 👤 Auteur

BEN-EL-AHMAR BADR — MSDE Édition 7
