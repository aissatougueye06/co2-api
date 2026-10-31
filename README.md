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
- [x] Intégration continue (GitHub Actions)


## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```


## Tests

```bash
ruff check . && ruff format --check . && pytest
```


## Structure

```
co2/                  le paquet
  __init__.py
  donnees.py          téléchargement vérifié des données
  preparation.py
  modele.py
  schemas.py
  api.py
scripts/
  telecharger_donnees.py
  entrainer.py
tests/                les tests
  test_donnees.py
  test_preparation.py
  test_modele.py
  test_api.py
data/                 non versionné, voir Données
  vehicules.csv
models/               non versionné, produit par entrainer.py
  co2.joblib
.github/workflows/    intégration continue
Dockerfile
.dockerignore
pyproject.toml        dépendances et configuration des outils
```


## Données

```bash
python scripts/telecharger_donnees.py
```

Télécharge l'archive ADEME, en extrait le CSV vers `data/vehicules.csv` et
vérifie son empreinte SHA-256. Si le fichier est déjà présent et valide, rien
n'est téléchargé ; si l'empreinte ne correspond pas, le script échoue sans rien
laisser en place.

`data/vehicules.csv` est le fichier `fic_etiq_edition_40-mars-2015.csv`
(ADEME, édition 40, mars 2015), renommé sans modification.

- Source : [archive ZIP](https://www.data.gouv.fr/api/1/datasets/r/bc42c2e3-d24c-4499-a966-d35656c6cfc1)
  du jeu de données [« Emissions de CO2 et de polluants des véhicules commercialisés en France »](https://www.data.gouv.fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france)
- Format : CSV, séparateur `;`, encodage ISO-8859-1
- SHA-256 du CSV extrait (pas de l'archive) :
  `06c6c0c765498f491b1b9921f85589a488083dbb5a62db3e5d685d2719442f2b`


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

```bash
docker build -t co2-api .
docker run --rm -p 8000:8000 co2-api
```

L'image se construit à partir du dépôt seul, en trois étapes : une base avec les
dépendances, une étape qui télécharge les données et entraîne le modèle, et
l'image finale qui ne reçoit que le modèle. Les données ne sont jamais livrées.
 
Le modèle est donc par construction celui que produit le code du dépôt, à partir
de données dont l'empreinte est vérifiée. Réentraîner impose de reconstruire
l'image.
 
Le port d'écoute se règle par la variable `PORT` (8000 par défaut), comme
l'attendent la plupart des hébergeurs.