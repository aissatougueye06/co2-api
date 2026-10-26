# co2-api

Mise en production du modèle de prédiction des émissions de CO₂ développé dans
[ml-co2-prediction](https://github.com/aissatougueye06/ml-co2-prediction).

Le notebook répondait à « quel modèle ? ». Ce dépôt répond à « comment le faire
tourner ailleurs que dans un notebook ? ».

## État

- [x] Préparation des données extraite en module testé
- [x] Entraînement, sérialisation et prédiction (`co2/modele.py`)

- [x] API de prédiction (FastAPI)
- [x] Conteneur Docker
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
  __init__.py
  preparation.py
  modele.py
  schemas.py
  api.py
scripts/
  entrainer.py
tests/                les tests
  test_preparation.py
  test_modele.py
  test_api.py
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


## Lancer l'API
 
```bash
uvicorn co2.api:app --reload
```
 
- `http://127.0.0.1:8000/docs` — documentation interactive, générée depuis les
  schémas Pydantic : on peut y envoyer une requête sans écrire une ligne de code.
- `http://127.0.0.1:8000/health` — sonde de disponibilité.
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"masse": 1300, "puissance": 90, "boite": "M 6"}'
# {"co2_g_par_km":119.2,"boite_connue":true}
```
 
Le chemin du modèle se surcharge par variable d'environnement (`CO2_MODELE`),
pour le déploiement. Sans modèle sur disque, le service démarre quand même :
`/health` répond `degrade` et `/predict` renvoie 503 — un service qui dit ce qui
lui manque se diagnostique mieux qu'un service qui refuse de démarrer.


## Conteneur Docker
 
Le modèle doit exister avant la construction : il est copié dans l'image.
 
```bash
python scripts/entrainer.py data/vehicules.csv   # produit models/co2.joblib
docker build -t co2-api .
docker run --rm -p 8000:8000 co2-api
```
 
L'API répond alors sur `http://localhost:8000` comme en local.
 
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"masse": 1300, "puissance": 90, "boite": "M 6"}'
```
 
Le modèle est embarqué dans l'image plutôt que monté au démarrage : l'image est
alors autonome et versionnée avec le modèle qu'elle sert — deux images, deux
modèles, aucune ambiguïté sur ce qui tourne. En contrepartie, réentraîner impose
de reconstruire l'image. C'est le bon arbitrage ici, où le modèle pèse quelques
kilo-octets et change rarement.