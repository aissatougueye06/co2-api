"""Entraînement, sérialisation et utilisation du modèle de prédiction du CO₂.

Le modèle retenu dans le notebook 02 : masse, puissance, puissance au carré et
type de boîte de vitesses, sur les véhicules essence.

Tout tient dans un `Pipeline` scikit-learn. Ce n'est pas une élégance : le
fichier sérialisé contient alors les transformations *et* les coefficients, donc
il est impossible de prédire avec un encodage différent de celui de
l'entraînement. C'est la première cause de bug en production.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

CIBLE = "co2_mixte"
CARACTERISTIQUES = ["masse_ordma_min", "puiss_max", "typ_boite_nb_rapp"]


@dataclass(frozen=True)
class Metriques:
    """Performance mesurée sur le jeu de test, et sa référence naïve."""

    mae: float
    r2: float
    mae_naive: float
    n_train: int
    n_test: int

    @property
    def rapport_au_naif(self) -> float:
        """De combien le modèle fait mieux qu'une prédiction constante."""
        return self.mae_naive / self.mae

    def __str__(self) -> str:
        return (
            f"MAE {self.mae:.1f} g/km (naif {self.mae_naive:.1f}, "
            f"x{self.rapport_au_naif:.1f}) | R2 {self.r2:.4f} | "
            f"{self.n_train} train / {self.n_test} test"
        )


def construire_modele() -> Pipeline:
    """Assemble les transformations et la régression en un seul objet.

    `handle_unknown="ignore"` fait qu'une boîte de vitesses jamais vue à
    l'entraînement produit des indicatrices nulles au lieu de lever une
    exception : une API ne doit pas tomber sur une valeur inattendue.
    """
    preparation = ColumnTransformer(
        [
            ("masse", "passthrough", ["masse_ordma_min"]),
            # degree=2 sur la seule puissance : produit puiss_max et puiss_max²
            (
                "puissance",
                PolynomialFeatures(degree=2, include_bias=False),
                ["puiss_max"],
            ),
            (
                "boite",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
                ["typ_boite_nb_rapp"],
            ),
        ]
    )
    return Pipeline([("preparation", preparation), ("regression", LinearRegression())])


def entrainer(
    d: pd.DataFrame, test_size: float = 0.2, random_state: int = 0
) -> tuple[Pipeline, Metriques]:
    """Entraîne le modèle et mesure sa performance hors échantillon.

    Retourne le modèle entraîné et ses métriques, jamais l'un sans l'autre :
    un modèle dont on ignore la performance ne doit pas circuler.
    """
    manquantes = set(CARACTERISTIQUES + [CIBLE]) - set(d.columns)
    if manquantes:
        raise ValueError(f"Colonnes absentes du jeu de données : {sorted(manquantes)}")

    X, y = d[CARACTERISTIQUES], d[CIBLE]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    modele = construire_modele().fit(X_tr, y_tr)
    prediction = modele.predict(X_te)
    naive = np.full(y_te.shape, y_tr.mean())

    metriques = Metriques(
        mae=mean_absolute_error(y_te, prediction),
        r2=r2_score(y_te, prediction),
        mae_naive=mean_absolute_error(y_te, naive),
        n_train=len(X_tr),
        n_test=len(X_te),
    )
    return modele, metriques


def sauvegarder(modele: Pipeline, chemin: str | Path) -> Path:
    """Sérialise le modèle entraîné. Crée le dossier parent si besoin."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, chemin)
    return chemin


def charger_modele(chemin: str | Path) -> Pipeline:
    """Recharge un modèle sérialisé."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin}. Lancer l'entraînement d'abord."
        )
    return joblib.load(chemin)


def predire(modele: Pipeline, masse: float, puissance: float, boite: str) -> float:
    """Prédit le CO₂ d'un véhicule, en g/km.

    Construit le DataFrame d'une ligne attendu par le pipeline : les noms de
    colonnes doivent être ceux de l'entraînement, d'où la constante partagée.
    """
    ligne = pd.DataFrame(
        [[masse, puissance, boite]],
        columns=CARACTERISTIQUES,
    )
    return float(modele.predict(ligne)[0])
