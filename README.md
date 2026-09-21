# co2-api

Mise en production du modèle de prédiction des émissions de CO₂ développé dans
[ml-co2-prediction](https://github.com/aissatougueye06/ml-co2-prediction).

Le notebook répondait à « quel modèle ? ». Ce dépôt répond à « comment le faire
tourner ailleurs que dans un notebook ? ».

## État

- [x] Préparation des données extraite en module testé
- [x] Entraînement, sérialisation et prédiction (`co2/modele.py`)

- [ ] API de prédiction (FastAPI)
- [ ] Conteneur Docker
- [ ] Intégration continue (GitHub Actions)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Structure

```
co2/                  le paquet
  preparation.py
  modele.py
scripts/
  entrainer.py
tests/                les tests
  test_preparation.py
  test_modele.py
data/                 non versionné, voir Données
  vehicules.csv
models/               non versionné, produit par entrainer.py
  co2.joblib
pyproject.toml        dépendances et configuration des outils
```

## Données

Les données ne sont pas versionnées (voir `.gitignore`). Pour les récupérer :

1. Télécharger la ressource « [2014] Emissions de polluants, CO2 et
   caractéristiques des véhicules commercialisés en France » depuis
   [data.gouv.fr](https://www.data.gouv.fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france)
2. L'enregistrer sous `data/vehicules.csv`


## Entraîner le modèle

```bash
python scripts/entrainer.py data/vehicules.csv
# 2406 vehicules ES apres preparation
# MAE 15.5 g/km (naif 42.9, x2.8) | R2 0.86 | ... train / ... test
# Modele ecrit dans models/co2.joblib
```

Le modèle est entraîné hors ligne et sérialisé. L'API se contentera de charger
`models/co2.joblib` : entraîner au démarrage d'un service rendrait chaque
redéploiement imprévisible.
