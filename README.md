# co2-api

Mise en production du modèle de prédiction des émissions de CO₂ développé dans
[ml-co2-prediction](https://github.com/aissatougueye06/ml-co2-prediction).

Le notebook répondait à « quel modèle ? ». Ce dépôt répond à « comment le faire
tourner ailleurs que dans un notebook ? ».

## État

- [x] Préparation des données extraite en module testé
- [ ] Entraînement et sérialisation du modèle
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
