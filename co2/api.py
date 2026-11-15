"""API de prédiction des émissions de CO₂.

    uvicorn co2.api:app --reload

Le modèle est chargé **une seule fois**, au démarrage, et gardé en mémoire.
Le recharger à chaque requête coûterait des centaines de millisecondes pour
rien ; l'entraîner au démarrage rendrait chaque redéploiement imprévisible
(cf. scripts/entrainer.py).
"""

import os
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from co2.modele import Artefact, charger_modele, predire
from co2.schemas import InfosModele, Prediction, Sante, Vehicule

CHEMIN_MODELE = Path(os.getenv("CO2_MODELE", "models/co2.joblib"))

# Rempli au démarrage par le lifespan. Un dict plutôt qu'une variable globale
# réassignée : les tests peuvent y injecter un artefact sans toucher au module.
etat: dict[str, Artefact | None] = {"artefact": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge l'artefact au démarrage, le libère à l'arrêt.

    Une erreur ici n'empêche pas le service de démarrer : /health répondra
    `modele_charge: false` et /predict renverra 503. Un service qui refuse de
    démarrer est plus difficile à diagnostiquer qu'un service qui dit ce qui
    lui manque.
    """
    try:
        etat["artefact"] = charger_modele(CHEMIN_MODELE)
    except (FileNotFoundError, TypeError):
        etat["artefact"] = None
    yield
    etat["artefact"] = None


app = FastAPI(
    title="API CO₂",
    description=(
        "Estime les émissions de CO₂ d'un véhicule thermique essence à partir de "
        "sa masse, sa puissance et son type de boîte. Modèle HistGradientBoosting "
        "entraîné sur les données ADEME 2014."
    ),
    # Une seule source de vérité : la version déclarée dans pyproject.toml.
    version=version("co2"),
    lifespan=lifespan,
)


def _artefact() -> Artefact:
    """L'artefact chargé, ou une 503 explicite s'il manque."""
    artefact = etat["artefact"]
    if artefact is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non chargé ({CHEMIN_MODELE}). Lancer l'entraînement.",
        )
    return artefact


def _boites_connues(modele: Pipeline) -> list[str]:
    """Les modalités de boîte vues à l'entraînement, lues dans l'encodeur."""
    encodeur = modele.named_steps["preparation"].named_transformers_["boite"]
    return sorted(encodeur.categories_[0])


@app.get("/health", response_model=Sante, tags=["technique"])
def sante() -> Sante:
    """Sonde de disponibilité, pour l'hébergeur et la supervision."""
    charge = etat["artefact"] is not None
    return Sante(statut="ok" if charge else "degrade", modele_charge=charge)


@app.get("/modele", response_model=InfosModele, tags=["technique"])
def infos_modele() -> InfosModele:
    """Ce que le modèle en service sait faire, et comment il a été mesuré."""
    artefact = _artefact()
    m = artefact.metriques
    return InfosModele(
        version=app.version,
        mae_g_par_km=round(m.mae, 2),
        mae_naive_g_par_km=round(m.mae_naive, 2),
        r2=round(m.r2, 4),
        n_test=m.n_test,
        domaine=artefact.domaine,
        boites_connues=_boites_connues(artefact.modele),
    )


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def prediction(vehicule: Vehicule) -> Prediction:
    """Estime le CO₂ d'un véhicule.

    Les bornes de plausibilité sont appliquées par Pydantic avant d'arriver
    ici : une masse hors plage ne touche jamais le modèle et reçoit un 422
    décrivant le champ fautif. Entre ces bornes et celles de l'entraînement,
    la prédiction est rendue mais signalée hors domaine.
    """
    artefact = _artefact()
    modele = artefact.modele
    return Prediction(
        co2_g_par_km=round(
            predire(modele, vehicule.masse, vehicule.puissance, vehicule.boite), 1
        ),
        boite_connue=vehicule.boite in _boites_connues(modele),
        dans_le_domaine=artefact.dans_le_domaine(vehicule.masse, vehicule.puissance),
    )
