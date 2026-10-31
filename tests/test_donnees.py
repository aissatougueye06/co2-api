"""Tests de co2.donnees.

Aucun accès réseau : `urllib` sait lire les URL `file://`, on sert donc une
archive ZIP locale comme si elle venait de data.gouv.fr.
"""

import hashlib
import zipfile

import pytest

from co2.donnees import EmpreinteInvalide, empreinte, telecharger

MEMBRE = "source.csv"
CONTENU = b"energ;masse_ordma_min;co2_mixte\nES;1000;104\n"
SHA256 = hashlib.sha256(CONTENU).hexdigest()


@pytest.fixture
def source(tmp_path):
    """Une archive « distante » contenant le CSV, et son URL."""
    chemin = tmp_path / "source.zip"
    with zipfile.ZipFile(chemin, "w") as zf:
        zf.writestr(MEMBRE, CONTENU)
    return chemin.as_uri()


def test_telecharge_extrait_et_verifie(source, tmp_path):
    destination = tmp_path / "data" / "vehicules.csv"
    telecharger(source, MEMBRE, destination, SHA256)
    assert destination.read_bytes() == CONTENU
    assert empreinte(destination) == SHA256


def test_empreinte_fausse_ne_laisse_aucun_fichier(source, tmp_path):
    """Le cas qui justifie tout le module : rien de douteux ne reste en place."""
    destination = tmp_path / "vehicules.csv"
    with pytest.raises(EmpreinteInvalide, match="Empreinte inattendue"):
        telecharger(source, MEMBRE, destination, "0" * 64)
    assert not destination.exists()
    assert not destination.with_suffix(".csv.part").exists()


def test_fichier_deja_present_et_valide_n_est_pas_retelecharge(tmp_path):
    destination = tmp_path / "vehicules.csv"
    destination.write_bytes(CONTENU)
    # URL inexistante : si la fonction tentait de télécharger, elle échouerait.
    telecharger((tmp_path / "absent.zip").as_uri(), MEMBRE, destination, SHA256)
    assert destination.read_bytes() == CONTENU


def test_reponse_qui_n_est_pas_un_zip_est_refusee(tmp_path):
    """Par exemple une page d'erreur HTML renvoyée à la place de l'archive."""
    page = tmp_path / "erreur.html"
    page.write_bytes(b"<html>Page introuvable</html>")
    destination = tmp_path / "vehicules.csv"
    with pytest.raises(EmpreinteInvalide, match="archive ZIP valide"):
        telecharger(page.as_uri(), MEMBRE, destination, SHA256)
    assert not destination.exists()


def test_csv_absent_de_l_archive_est_refuse(source, tmp_path):
    """Par exemple si l'ADEME renomme le fichier dans une nouvelle archive."""
    destination = tmp_path / "vehicules.csv"
    with pytest.raises(EmpreinteInvalide, match="absent de l'archive"):
        telecharger(source, "autre.csv", destination, SHA256)
    assert not destination.exists()
