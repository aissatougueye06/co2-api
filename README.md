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
co2/            le paquet
  preparation.py
tests/          les tests
pyproject.toml  dependances et configuration des outils
```

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
