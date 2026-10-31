"""Téléchargement vérifié des données ADEME.

Les données ne sont pas versionnées : on les récupère à la source, et on vérifie
leur empreinte SHA-256. Sans cette vérification, une mise à jour silencieuse du
fichier sur data.gouv.fr changerait le modèle entraîné sans que rien ne le
signale — l'image se construirait, les tests passeraient, et le modèle en
production ne serait plus celui qu'on a évalué.
"""

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Ressource ADEME sur data.gouv.fr : une archive ZIP contenant un seul CSV
# (édition 40, mars 2015, séparateur « ; », encodage ISO-8859-1).
URL_ADEME_2015 = (
    "https://www.data.gouv.fr/api/1/datasets/r/bc42c2e3-d24c-4499-a966-d35656c6cfc1"
)
CSV_ADEME_2015 = "fic_etiq_edition_40-mars-2015.csv"
# Empreinte du CSV extrait, pas de l'archive : le ZIP peut être recompressé
# sans que les données changent.
SHA256_ADEME_2015 = "06c6c0c765498f491b1b9921f85589a488083dbb5a62db3e5d685d2719442f2b"


class EmpreinteInvalide(Exception):
    """Le fichier téléchargé n'est pas celui attendu."""


def empreinte(chemin: str | Path) -> str:
    """SHA-256 d'un fichier, lu par blocs pour ne pas tout charger en mémoire."""
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def telecharger(url: str, membre: str, destination: str | Path, sha256: str) -> Path:
    """Télécharge l'archive ZIP `url`, en extrait `membre` vers `destination`
    et vérifie l'empreinte du fichier extrait.

    Si le fichier existe déjà avec la bonne empreinte, rien n'est téléchargé.

    L'écriture passe par un fichier temporaire renommé seulement une fois
    l'empreinte vérifiée : un téléchargement interrompu ou corrompu ne laisse
    jamais de faux fichier en place.
    """
    destination = Path(destination)
    if destination.exists() and empreinte(destination) == sha256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporaire = destination.with_suffix(destination.suffix + ".part")
    with tempfile.TemporaryDirectory() as dossier:
        archive = Path(dossier) / "archive.zip"
        try:
            urllib.request.urlretrieve(url, archive)
            with (
                zipfile.ZipFile(archive) as zf,
                zf.open(membre) as src,
                open(temporaire, "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile as erreur:
            raise EmpreinteInvalide(
                f"{url} n'a pas renvoyé une archive ZIP valide"
            ) from erreur
        except KeyError as erreur:
            raise EmpreinteInvalide(f"{membre} absent de l'archive {url}") from erreur

        try:
            obtenue = empreinte(temporaire)
            if obtenue != sha256:
                raise EmpreinteInvalide(
                    f"Empreinte inattendue pour {membre} ({url})\n"
                    f"  attendue : {sha256}\n"
                    f"  obtenue  : {obtenue}\n"
                    "Le fichier source a peut-être changé : vérifier avant de "
                    "mettre à jour l'empreinte."
                )
            temporaire.replace(destination)
        finally:
            temporaire.unlink(missing_ok=True)
    return destination
