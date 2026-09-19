"""Tests de co2.modele.

Jeu synthétique construit à partir d'une relation connue : le test vérifie la
mécanique (le modèle s'entraîne, se sauvegarde, se recharge, prédit), pas la
qualité de l'ajustement sur les vraies données ADEME.
"""

import numpy as np
import pandas as pd
import pytest

from co2.modele import (
    CARACTERISTIQUES,
    charger_modele,
    entrainer,
    predire,
    sauvegarder,
)


@pytest.fixture
def jeu() -> pd.DataFrame:
    """80 véhicules dont le CO₂ suit une relation connue, avec un peu de bruit."""
    generateur = np.random.default_rng(0)
    n = 80
    masse = generateur.uniform(900, 2200, n)
    puissance = generateur.uniform(50, 400, n)
    boite = generateur.choice(["M 5", "M 6", "A 7", "A 8"], n)

    co2 = (
        40
        + 0.05 * masse
        + 0.20 * puissance
        + 0.0002 * puissance**2
        + generateur.normal(0, 5, n)
    )
    return pd.DataFrame(
        {
            "masse_ordma_min": masse,
            "puiss_max": puissance,
            "typ_boite_nb_rapp": boite,
            "co2_mixte": co2,
        }
    )


def test_entrainement_bat_la_reference_naive(jeu):
    """Un modèle qui ne bat pas la prédiction constante n'a rien appris."""
    _, metriques = entrainer(jeu)
    assert metriques.mae < metriques.mae_naive
    assert metriques.rapport_au_naif > 2
    assert metriques.n_train + metriques.n_test == len(jeu)


def test_colonne_manquante_echoue_explicitement(jeu):
    """Mieux vaut une erreur nommée qu'un KeyError au milieu du pipeline."""
    with pytest.raises(ValueError, match="puiss_max"):
        entrainer(jeu.drop(columns=["puiss_max"]))


def test_aller_retour_sur_disque(jeu, tmp_path):
    """Le modèle rechargé doit prédire exactement comme l'original.

    C'est le test qui protège de la dérive entraînement/service : si la
    sérialisation perdait une étape de préparation, les prédictions
    divergeraient ici.
    """
    modele, _ = entrainer(jeu)
    chemin = sauvegarder(modele, tmp_path / "modeles" / "co2.joblib")
    assert chemin.exists()

    recharge = charger_modele(chemin)
    echantillon = jeu[CARACTERISTIQUES].head(10)
    np.testing.assert_allclose(
        modele.predict(echantillon), recharge.predict(echantillon)
    )


def test_modele_absent_echoue_explicitement(tmp_path):
    with pytest.raises(FileNotFoundError, match="Lancer l'entraînement"):
        charger_modele(tmp_path / "jamais-cree.joblib")


def test_predire_renvoie_un_flottant_plausible(jeu):
    modele, _ = entrainer(jeu)
    valeur = predire(modele, masse=1300, puissance=90, boite="M 6")
    assert isinstance(valeur, float)
    assert 50 < valeur < 400


def test_boite_inconnue_ne_fait_pas_tomber_le_modele(jeu):
    """Une API reçoit des valeurs inattendues : elle ne doit pas planter.

    scikit-learn avertit et encode la modalité inconnue en zéros. La prédiction
    revient donc à celle de la modalité de référence — dégradée, mais rendue.
    """
    modele, _ = entrainer(jeu)
    with pytest.warns(UserWarning, match="unknown categories"):
        valeur = predire(modele, masse=1300, puissance=90, boite="X 42")
    assert isinstance(valeur, float)


def test_la_puissance_augmente_le_co2(jeu):
    """Vérification de bon sens : à masse et boîte égales, plus de puissance émet plus."""
    modele, _ = entrainer(jeu)
    faible = predire(modele, masse=1300, puissance=70, boite="M 6")
    forte = predire(modele, masse=1300, puissance=300, boite="M 6")
    assert forte > faible
