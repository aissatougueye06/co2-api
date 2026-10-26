# Image de base « slim » : Debian minimal + Python, ~10x plus légère que l'image
# complète. Version figée (3.12) : « python:3-slim » changerait de version majeure
# sans prévenir, un jour, au milieu d'un déploiement.
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE : pas de .pyc, inutiles dans un conteneur jetable.
# PYTHONUNBUFFERED : les logs sortent immédiatement au lieu d'être tamponnés —
# sans ça, les messages d'une application qui plante peuvent ne jamais s'afficher.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Couche 1 : les dépendances ---------------------------------------------
# On copie le pyproject SEUL, puis on installe. Docker met chaque étape en cache
# et ne rejoue que celles dont les fichiers ont changé : tant que les dépendances
# bougent pas, cette couche (la plus lente, ~1 min) est réutilisée telle quelle.
# Copier tout le code d'abord invaliderait le cache à chaque ligne modifiée.
COPY pyproject.toml README.md ./
COPY co2/__init__.py ./co2/
RUN pip install --no-cache-dir .

# --- Couche 2 : le code et le modèle ----------------------------------------
# Change souvent, donc arrive après : seule cette couche est reconstruite.
COPY co2/ ./co2/
COPY models/ ./models/

# Exécuter en root dans un conteneur est un défaut de sécurité classique : une
# faille de l'application donnerait les pleins pouvoirs sur le conteneur.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

# La sonde construite dans l'API sert ici : Docker (et l'hébergeur) savent quand
# le service est réellement prêt, pas seulement quand le processus est lancé.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# --host 0.0.0.0 est OBLIGATOIRE : par défaut uvicorn n'écoute que 127.0.0.1,
# c'est-à-dire l'intérieur du conteneur — le service serait injoignable.
# Pas de --reload : c'est un outil de développement.
CMD ["uvicorn", "co2.api:app", "--host", "0.0.0.0", "--port", "8000"]
