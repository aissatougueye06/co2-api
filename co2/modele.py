"""Entraînement, sérialisation et utilisation du modèle de prédiction du CO₂.

Le modèle retenu dans le notebook 04 : HistGradientBoosting sur la masse, la
puissance et le type de boîte, pour les véhicules essence.

Deux choix viennent de ce notebook et conditionnent tout le reste :

- **Découpage par configuration.** Beaucoup de véhicules partagent exactement la
  même masse, la même puissance et la même boîte. Un découpage aléatoire en place
  des « jumeaux » des deux côtés, et un modèle à base d'arbres les apprend par
  cœur : la mesure est flattée, d'autant plus que le modèle sait mémoriser. On
  découpe donc par configuration (`GroupShuffleSplit`).
- **Domaine de validité.** Un arbre n'extrapole pas : au-delà des valeurs vues à
  l'entraînement, sa prédiction reste plate. Les bornes vues sont enregistrées
  avec le modèle pour que l'API puisse le signaler.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CIBLE = "co2_mixte"
CARACTERISTIQUES = ["masse_ordma_min", "puiss_max", "typ_boite_nb_rapp"]
NUMERIQUES = ["masse_ordma_min", "puiss_max"]


@dataclass(frozen=True)
class Metriques:
    """Performance mesurée sur des configurations absentes de l'entraînement."""

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
            f"MAE {self.mae:.2f} g/km (naif {self.mae_naive:.1f}, "
            f"x{self.rapport_au_naif:.1f}) | R2 {self.r2:.4f} | "
            f"{self.n_train} train / {self.n_test} test"
        )


@dataclass(frozen=True)
class Artefact:
    """Ce qui est sérialisé : le modèle, et ce qu'il faut savoir pour s'y fier.

    Le pipeline seul ne dit ni comment il a été mesuré, ni sur quel domaine il
    est valide. Les garder ensemble évite qu'ils se séparent.
    """

    modele: Pipeline
    metriques: Metriques
    domaine: dict[str, tuple[float, float]]

    def dans_le_domaine(self, masse: float, puissance: float) -> bool:
        """Vrai si le véhicule est dans les bornes vues à l'entraînement."""
        valeurs = {"masse_ordma_min": masse, "puiss_max": puissance}
        return all(
            bas <= valeurs[col] <= haut for col, (bas, haut) in self.domaine.items()
        )


def configuration(X: pd.DataFrame) -> pd.Series:
    """Clé de groupement : deux lignes de même clé sont indiscernables du modèle.

    Même construction que dans le notebook 04, pour que les découpages coïncident.
    """
    return X[CARACTERISTIQUES].astype(str).agg("|".join, axis=1)


def decouper(
    d: pd.DataFrame, test_size: float = 0.2, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare entraînement et test sans qu'une configuration soit des deux côtés."""
    separateur = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    idx_tr, idx_te = next(separateur.split(d, groups=configuration(d)))
    return d.iloc[idx_tr], d.iloc[idx_te]


def construire_modele() -> Pipeline:
    """Le pipeline retenu dans le notebook 04, hyperparamètres compris.

    Ni `PolynomialFeatures` ni normalisation : un arbre trouve seul la courbure
    et coupe sur des seuils indépendants de l'échelle. `unknown_value=-1` fait
    qu'une boîte jamais vue produit un code hors des valeurs connues au lieu de
    lever une exception.
    """
    preparation = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIQUES),
            (
                "boite",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["typ_boite_nb_rapp"],
            ),
        ]
    )
    estimateur = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_iter=600,
        max_leaf_nodes=7,
        min_samples_leaf=3,
        early_stopping=False,
        random_state=0,
    )
    return Pipeline([("preparation", preparation), ("estimateur", estimateur)])


def entrainer(
    d: pd.DataFrame, test_size: float = 0.2, random_state: int = 0
) -> Artefact:
    """Mesure le modèle sur des configurations inédites, puis le réentraîne sur tout.

    Les métriques viennent d'un modèle entraîné sur une partie des données et
    évalué sur le reste. Le modèle livré est ensuite réentraîné sur l'ensemble,
    avec les mêmes hyperparamètres : il a vu plus de données, ses métriques sont
    donc une estimation légèrement prudente.
    """
    manquantes = set(CARACTERISTIQUES + [CIBLE]) - set(d.columns)
    if manquantes:
        raise ValueError(f"Colonnes absentes du jeu de données : {sorted(manquantes)}")

    d_tr, d_te = decouper(d, test_size, random_state)
    mesure = construire_modele().fit(d_tr[CARACTERISTIQUES], d_tr[CIBLE])
    prediction = mesure.predict(d_te[CARACTERISTIQUES])
    naive = np.full(len(d_te), d_tr[CIBLE].mean())

    metriques = Metriques(
        mae=mean_absolute_error(d_te[CIBLE], prediction),
        r2=r2_score(d_te[CIBLE], prediction),
        mae_naive=mean_absolute_error(d_te[CIBLE], naive),
        n_train=len(d_tr),
        n_test=len(d_te),
    )
    modele = construire_modele().fit(d[CARACTERISTIQUES], d[CIBLE])
    domaine = {col: (float(d[col].min()), float(d[col].max())) for col in NUMERIQUES}
    return Artefact(modele=modele, metriques=metriques, domaine=domaine)


def sauvegarder(artefact: Artefact, chemin: str | Path) -> Path:
    """Sérialise l'artefact. Crée le dossier parent si besoin."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, chemin)
    return chemin


def charger_modele(chemin: str | Path) -> Artefact:
    """Recharge un artefact sérialisé."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin}. Lancer l'entraînement d'abord."
        )
    artefact = joblib.load(chemin)
    if not isinstance(artefact, Artefact):
        raise TypeError(
            f"{chemin} n'est pas un Artefact (ancien format ?). Réentraîner."
        )
    return artefact


def predire(modele: Pipeline, masse: float, puissance: float, boite: str) -> float:
    """Prédit le CO₂ d'un véhicule, en g/km."""
    ligne = pd.DataFrame([[masse, puissance, boite]], columns=CARACTERISTIQUES)
    return float(modele.predict(ligne)[0])
