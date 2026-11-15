"""Tests de co2.modele.

Jeu synthétique construit à partir d'une relation connue : les tests vérifient
la mécanique (découpage, entraînement, sérialisation, domaine), pas la qualité
de l'ajustement sur les vraies données ADEME.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from co2.modele import (
    CARACTERISTIQUES,
    Artefact,
    charger_modele,
    configuration,
    construire_modele,
    decouper,
    entrainer,
    predire,
    sauvegarder,
)


@pytest.fixture(scope="module")
def jeu() -> pd.DataFrame:
    """60 configurations présentes chacune deux fois, avec des CO₂ différents.

    Les doublons de configuration sont voulus : ce sont eux que le découpage
    doit garder du même côté.
    """
    generateur = np.random.default_rng(0)
    n = 60
    configs = pd.DataFrame(
        {
            "masse_ordma_min": generateur.uniform(900, 2200, n).round(),
            "puiss_max": generateur.uniform(50, 400, n).round(),
            "typ_boite_nb_rapp": generateur.choice(["M 5", "M 6", "A 7", "A 8"], n),
        }
    )
    d = pd.concat([configs, configs], ignore_index=True)
    d["co2_mixte"] = (
        40
        + 0.05 * d["masse_ordma_min"]
        + 0.20 * d["puiss_max"]
        + 0.0002 * d["puiss_max"] ** 2
        + generateur.normal(0, 5, len(d))
    )
    return d


@pytest.fixture(scope="module")
def artefact(jeu) -> Artefact:
    """Entraîné une fois pour tout le module : 600 itérations par test, c'est lent.

    Partageable parce qu'aucun test ne le modifie (`Artefact` est figé).
    """
    return entrainer(jeu)


def test_decouper_ne_partage_aucune_configuration(jeu):
    """La garantie qui protège la mesure : aucun jumeau des deux côtés."""
    d_tr, d_te = decouper(jeu)
    assert set(configuration(d_tr)).isdisjoint(configuration(d_te))
    assert len(d_tr) + len(d_te) == len(jeu)


def test_entrainement_bat_la_reference_naive(jeu, artefact):
    """Un modèle qui ne bat pas la prédiction constante n'a rien appris."""
    metriques = artefact.metriques
    assert metriques.mae < metriques.mae_naive
    assert metriques.rapport_au_naif > 2
    assert metriques.n_train + metriques.n_test == len(jeu)


def test_colonne_manquante_echoue_explicitement(jeu):
    """Mieux vaut une erreur nommée qu'un KeyError au milieu du pipeline."""
    with pytest.raises(ValueError, match="puiss_max"):
        entrainer(jeu.drop(columns=["puiss_max"]))


def test_domaine_enregistre_les_bornes_d_entrainement(jeu, artefact):
    bas, haut = artefact.domaine["puiss_max"]
    assert (bas, haut) == (jeu["puiss_max"].min(), jeu["puiss_max"].max())

    masse = float(jeu["masse_ordma_min"].median())
    assert artefact.dans_le_domaine(masse, haut), "la borne elle-même est incluse"
    assert not artefact.dans_le_domaine(masse, haut + 1)
    assert not artefact.dans_le_domaine(jeu["masse_ordma_min"].min() - 1, bas)


def test_pas_d_extrapolation_au_dela_du_domaine(jeu, artefact):
    """La propriété des arbres qui justifie `dans_le_domaine` dans l'API."""
    haut = artefact.domaine["puiss_max"][1]
    a = predire(artefact.modele, masse=1500, puissance=haut + 100, boite="M 6")
    b = predire(artefact.modele, masse=1500, puissance=haut + 500, boite="M 6")
    assert a == b


def test_aller_retour_sur_disque(jeu, artefact, tmp_path):
    """L'artefact rechargé doit prédire exactement comme l'original."""
    chemin = sauvegarder(artefact, tmp_path / "modeles" / "co2.joblib")
    recharge = charger_modele(chemin)

    echantillon = jeu[CARACTERISTIQUES].head(10)
    np.testing.assert_allclose(
        artefact.modele.predict(echantillon), recharge.modele.predict(echantillon)
    )
    assert recharge.metriques == artefact.metriques
    assert recharge.domaine == artefact.domaine


def test_ancien_format_refuse_explicitement(jeu, tmp_path):
    """Un pipeline nu (format d'avant l'artefact) ne doit pas être servi."""
    chemin = tmp_path / "ancien.joblib"
    joblib.dump(
        construire_modele().fit(jeu[CARACTERISTIQUES], jeu["co2_mixte"]), chemin
    )
    with pytest.raises(TypeError, match="Réentraîner"):
        charger_modele(chemin)


def test_modele_absent_echoue_explicitement(tmp_path):
    with pytest.raises(FileNotFoundError, match="Lancer l'entraînement"):
        charger_modele(tmp_path / "jamais-cree.joblib")


def test_predire_renvoie_un_flottant_plausible(jeu, artefact):
    valeur = predire(artefact.modele, masse=1300, puissance=90, boite="M 6")
    assert isinstance(valeur, float)
    assert 50 < valeur < 400


def test_boite_inconnue_ne_fait_pas_tomber_le_modele(jeu, artefact):
    """Une API reçoit des valeurs inattendues : elle ne doit pas planter."""
    valeur = predire(artefact.modele, masse=1300, puissance=90, boite="X 42")
    assert isinstance(valeur, float)


def test_la_puissance_augmente_le_co2(jeu, artefact):
    """Bon sens : à masse et boîte égales, nettement plus de puissance émet plus."""
    modele = artefact.modele
    faible = predire(modele, masse=1300, puissance=70, boite="M 6")
    forte = predire(modele, masse=1300, puissance=300, boite="M 6")
    assert forte > faible


def test_artefact_est_bien_le_type_serialise(artefact, tmp_path):
    chemin = sauvegarder(artefact, tmp_path / "co2.joblib")
    assert isinstance(joblib.load(chemin), Artefact)
