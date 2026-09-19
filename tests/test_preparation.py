"""Tests de co2.preparation.

Les tests utilisent un jeu de données synthétique, pas le CSV ADEME : la logique
de préparation doit être vérifiable sans fichier, en quelques millisecondes, et
avec des cas choisis pour être lisibles.
"""

import numpy as np
import pandas as pd
import pytest

from co2.preparation import (
    ESSENTIELLES,
    lignes_exploitables,
    preparer,
    taux_de_replicats,
)


@pytest.fixture
def brut() -> pd.DataFrame:
    """Six lignes couvrant les trois comportements attendus de `preparer`.

    - PEUGEOT 208 : deux variantes strictement identiques -> un seul véhicule
    - PEUGEOT 208 (1090 kg) : finition plus lourde -> véhicule distinct
    - RENAULT CLIO : masse manquante -> ligne écartée
    - VOLKSWAGEN GOLF : diesel -> hors du filtre essence
    """
    return pd.DataFrame(
        {
            "energ": ["ES", "ES", "ES", "ES", "GO", "ES"],
            "lib_mrq_doss": [
                "PEUGEOT",
                "PEUGEOT",
                "PEUGEOT",
                "RENAULT",
                "VOLKSWAGEN",
                "FERRARI",
            ],
            "lib_mod_doss": ["208", "208", "208", "CLIO", "GOLF", "F12"],
            "typ_boite_nb_rapp": ["M 5", "M 5", "M 5", "A 6", "M 6", "A 7"],
            "masse_ordma_min": [1000.0, 1000.0, 1090.0, np.nan, 1300.0, 1700.0],
            "puiss_max": [60.0, 60.0, 60.0, 70.0, 81.0, 544.0],
            "co2_mixte": [104.0, 104.0, 110.0, 120.0, 99.0, 350.0],
        }
    )


def test_lignes_exploitables_ne_dedoublonne_pas(brut):
    """Etape intermediaire : les NaN sont ecartes, les variantes conservees."""
    assert len(lignes_exploitables(brut, "ES")) == 4
    assert len(preparer(brut, "ES")) == 3


def test_deduplique_les_variantes_identiques(brut):
    """Deux déclarations identiques comptent pour un seul véhicule."""
    d = preparer(brut, "ES")
    p208 = d[(d["lib_mrq_doss"] == "PEUGEOT") & (d["lib_mod_doss"] == "208")]
    assert len(p208) == 2, "les deux finitions de masses différentes doivent survivre"
    assert sorted(p208["masse_ordma_min"]) == [1000.0, 1090.0]


def test_ecarte_les_lignes_sans_valeur_essentielle(brut):
    """Une ligne sans masse, puissance ou CO₂ est inutilisable."""
    d = preparer(brut, "ES")
    assert "CLIO" not in d["lib_mod_doss"].values
    assert d[ESSENTIELLES].notna().all().all()


def test_filtre_sur_la_motorisation_et_ajoute_le_terme_quadratique(brut):
    """Seule la motorisation demandée est retenue, et `puiss_max²` est calculée."""
    essence = preparer(brut, "ES")
    assert "GOLF" not in essence["lib_mod_doss"].values
    assert len(essence) == 3  # 208 (x2 finitions) + F12

    diesel = preparer(brut, "GO")
    assert diesel["lib_mod_doss"].tolist() == ["GOLF"]

    assert (essence["puiss_max²"] == essence["puiss_max"] ** 2).all()


def test_taux_de_replicats(brut):
    """4 lignes essence exploitables -> 3 véhicules, soit 25 % de réplicats."""
    assert taux_de_replicats(brut, "ES") == pytest.approx(0.25)


def test_motorisation_hors_perimetre(brut):
    """Un code hors périmètre échoue explicitement plutôt que de renvoyer du vide."""
    with pytest.raises(ValueError, match="hors périmètre"):
        preparer(brut, "EL")


def test_ne_modifie_pas_le_dataframe_d_entree(brut):
    """`preparer` ne doit avoir aucun effet de bord sur son argument."""
    avant = brut.copy()
    preparer(brut, "ES")
    pd.testing.assert_frame_equal(brut, avant)
