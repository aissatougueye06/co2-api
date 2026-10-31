# Construction en trois étapes, pour que l'image se construise à partir du dépôt
# seul — sur ton Mac, dans la CI ou chez l'hébergeur — sans dépendre d'un
# fichier présent uniquement sur une machine.
#
#   base         Python + dépendances, partagée par les deux suivantes
#   entrainement télécharge les données, entraîne, produit models/co2.joblib
#   service      l'image finale : ne reçoit QUE le modèle, pas les données
#
# Seule la dernière étape est livrée. Les données et les outils d'entraînement
# restent dans l'étape intermédiaire, jetée après la construction.

# --- base ---------------------------------------------------------------------
# Version figée (3.12) : « python:3-slim » changerait de version majeure sans
# prévenir. C'est aussi la version testée par la CI.
FROM python:3.12-slim AS base

# PYTHONDONTWRITEBYTECODE : pas de .pyc, inutiles dans un conteneur jetable.
# PYTHONUNBUFFERED : les logs sortent immédiatement, sans tampon — sinon les
# messages d'une application qui plante peuvent ne jamais s'afficher.
# PYTHONPATH : le paquet installé à l'étape base ne contient que __init__.py
# (pour le cache des dépendances) ; /app/co2, copié ensuite, doit passer avant.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Les dépendances d'abord, seules : tant qu'elles ne changent pas, Docker
# réutilise cette couche (la plus lente) au lieu de tout réinstaller.
COPY pyproject.toml README.md ./
COPY co2/__init__.py ./co2/
RUN pip install --no-cache-dir .

# --- entrainement -------------------------------------------------------------
FROM base AS entrainement

COPY co2/ ./co2/
COPY scripts/ ./scripts/

# Téléchargement vérifié par empreinte SHA-256 (cf. co2/donnees.py), puis
# entraînement. Si data.gouv.fr change le fichier, la construction échoue au
# lieu de produire silencieusement un autre modèle.
RUN python scripts/telecharger_donnees.py --sortie data/vehicules.csv \
 && python scripts/entrainer.py data/vehicules.csv --sortie models/co2.joblib

# --- service ------------------------------------------------------------------
FROM base AS service

COPY co2/ ./co2/
# Le seul transfert entre étapes : le modèle produit ci-dessus.
COPY --from=entrainement /app/models/co2.joblib ./models/co2.joblib

# Pas de root : une faille de l'application ne donnerait pas les pleins pouvoirs.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# Les hébergeurs imposent souvent leur port via la variable PORT. 8000 sinon.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

# `sh -c` pour que ${PORT} soit lu au démarrage (la forme JSON ne remplace pas
# les variables). `exec` pour qu'uvicorn REMPLACE le shell : c'est alors lui qui
# reçoit le signal d'arrêt de l'hébergeur et s'arrête proprement, au lieu d'être
# tué brutalement après un délai.
CMD ["sh", "-c", "exec uvicorn co2.api:app --host 0.0.0.0 --port ${PORT}"]
