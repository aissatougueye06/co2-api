"""Tests de co2.api.

`TestClient` appelle l'application sans lancer de serveur : les tests sont
aussi rapides que les autres et ne dépendent d'aucun port libre.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from co2.api import app, etat
from co2.modele import Artefact, entrainer

VEHICULE = {"masse": 1300, "puissance": 90, "boite": "M 6"}


@pytest.fixture(scope="module")
def artefact() -> Artefact:
    """Entraîné une fois pour tout le module : réentraîner à chaque test est lent."""
    generateur = np.random.default_rng(0)
    n = 80
    masse = generateur.uniform(900, 2200, n)
    puissance = generateur.uniform(50, 400, n)
    jeu = pd.DataFrame(
        {
            "masse_ordma_min": masse,
            "puiss_max": puissance,
            "typ_boite_nb_rapp": generateur.choice(["M 5", "M 6", "A 7", "A 8"], n),
            "co2_mixte": (
                40
                + 0.05 * masse
                + 0.20 * puissance
                + 0.0002 * puissance**2
                + generateur.normal(0, 5, n)
            ),
        }
    )
    return entrainer(jeu)


@pytest.fixture
def client(artefact):
    """Client sur une application dont l'artefact est injecté en mémoire.

    Recréé à chaque test, car certains tests modifient `etat`. On injecte plutôt
    que de sérialiser sur disque : le test porte sur les routes, pas sur le
    chargement du fichier.
    """
    with TestClient(app) as c:
        etat["artefact"] = artefact
        yield c


def test_health_signale_le_modele_charge(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"statut": "ok", "modele_charge": True}


def test_prediction_plausible(client):
    reponse = client.post("/predict", json=VEHICULE)
    assert reponse.status_code == 200
    corps = reponse.json()
    assert 50 < corps["co2_g_par_km"] < 400
    assert corps["boite_connue"] is True
    assert corps["dans_le_domaine"] is True


@pytest.mark.parametrize(
    ("champ", "valeur"),
    [
        ("masse", -1600),
        ("masse", 0),
        ("masse", 9000),
        ("puissance", -10),
        ("puissance", 0),
        ("boite", ""),
    ],
)
def test_valeur_implausible_rejetee_avant_le_modele(client, champ, valeur):
    """Le cœur de la validation : ces requêtes n'atteignent jamais le modèle.

    Sans Pydantic, le modèle rendrait une valeur pour une masse négative,
    sans le moindre avertissement.
    """
    reponse = client.post("/predict", json={**VEHICULE, champ: valeur})
    assert reponse.status_code == 422
    assert reponse.json()["detail"][0]["loc"][-1] == champ


def test_champ_manquant_rejete(client):
    reponse = client.post("/predict", json={"masse": 1300, "puissance": 90})
    assert reponse.status_code == 422


def test_boite_inconnue_est_signalee_pas_refusee(client):
    """Une modalité jamais vue dégrade la prédiction sans faire tomber l'API."""
    reponse = client.post("/predict", json={**VEHICULE, "boite": "X 42"})
    assert reponse.status_code == 200
    assert reponse.json()["boite_connue"] is False


def test_hors_domaine_est_signale_pas_refuse(client):
    """Entre les bornes de Pydantic et celles de l'entraînement : rendu, signalé."""
    reponse = client.post("/predict", json={**VEHICULE, "puissance": 900})
    assert reponse.status_code == 200
    assert reponse.json()["dans_le_domaine"] is False


def test_modele_decrit_ce_qui_est_en_service(client):
    corps = client.get("/modele").json()
    assert corps["version"] == app.version
    assert corps["mae_g_par_km"] < corps["mae_naive_g_par_km"]
    assert set(corps["domaine"]) == {"masse_ordma_min", "puiss_max"}
    assert "M 6" in corps["boites_connues"]


def test_sans_modele_les_routes_renvoient_503(client):
    etat["artefact"] = None
    assert client.get("/health").json()["modele_charge"] is False
    for reponse in (client.post("/predict", json=VEHICULE), client.get("/modele")):
        assert reponse.status_code == 503
        assert "Modèle non chargé" in reponse.json()["detail"]
