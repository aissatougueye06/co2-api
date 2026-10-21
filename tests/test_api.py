"""Tests de co2.api.

`TestClient` appelle l'application sans lancer de serveur : les tests sont
aussi rapides que les autres et ne dépendent d'aucun port libre.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from co2.api import app, etat
from co2.modele import entrainer

VEHICULE = {"masse": 1300, "puissance": 90, "boite": "M 6"}


@pytest.fixture
def client():
    """Client sur une application dont le modèle est entraîné en mémoire.

    On injecte dans `etat` plutôt que de sérialiser un modèle sur disque :
    le test porte sur les routes, pas sur le chargement du fichier.
    """
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
    modele, _ = entrainer(jeu)
    with TestClient(app) as c:
        etat["modele"] = modele
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

    Sans Pydantic, `predire(masse=-1600, ...)` rend -30 g/km sans avertissement.
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


def test_sans_modele_la_prediction_renvoie_503(client):
    etat["modele"] = None
    assert client.get("/health").json()["modele_charge"] is False
    reponse = client.post("/predict", json=VEHICULE)
    assert reponse.status_code == 503
    assert "Modèle non chargé" in reponse.json()["detail"]
