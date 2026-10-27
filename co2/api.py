"""API de prédiction des émissions de CO₂.

    uvicorn co2.api:app --reload

Le modèle est chargé **une seule fois**, au démarrage, et gardé en mémoire.
Le recharger à chaque requête coûterait des centaines de millisecondes pour
rien ; l'entraîner au démarrage rendrait chaque redéploiement imprévisible
(cf. scripts/entrainer.py).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from co2.modele import charger_modele, predire
from co2.schemas import Prediction, Sante, Vehicule

CHEMIN_MODELE = Path(os.getenv("CO2_MODELE", "models/co2.joblib"))

# Rempli au démarrage par le lifespan. Un dict plutôt qu'une variable globale
# réassignée : les tests peuvent y injecter un modèle sans toucher au module.
etat: dict[str, Pipeline | None] = {"modele": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, le libère à l'arrêt.

    Une erreur ici n'empêche pas le service de démarrer : /health répondra
    `modele_charge: false` et /predict renverra 503. Un service qui refuse de
    démarrer est plus difficile à diagnostiquer qu'un service qui dit ce qui
    lui manque.
    """
    try:
        etat["modele"] = charger_modele(CHEMIN_MODELE)
    except FileNotFoundError:
        etat["modele"] = None
    yield
    etat["modele"] = None  # remis à None après le yield, c.-à-d. à l’arrêt.


app = FastAPI(
    title="API CO₂",
    description=(
        "Estime les émissions de CO₂ d'un véhicule thermique à partir de sa "
        "masse, sa puissance et son type de boîte. Modèle entraîné sur les "
        "données ADEME 2014."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _modele() -> Pipeline:
    """Le modèle chargé, ou une 503 explicite s'il manque."""
    modele = etat["modele"]
    if modele is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non chargé ({CHEMIN_MODELE}). Lancer l'entraînement.",
        )
    return modele


def _boites_connues(modele: Pipeline) -> set[str]:
    """Les modalités de boîte vues à l'entraînement, lues dans l'encodeur."""
    encodeur = modele.named_steps["preparation"].named_transformers_["boite"]
    return set(encodeur.categories_[0])


@app.get("/health", response_model=Sante, tags=["technique"])
def sante() -> Sante:
    """Sonde de disponibilité, pour l'hébergeur et la supervision."""
    charge = etat["modele"] is not None
    return Sante(statut="ok" if charge else "degrade", modele_charge=charge)


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def prediction(vehicule: Vehicule) -> Prediction:
    """Estime le CO₂ d'un véhicule.

    Les bornes de plausibilité sont appliquées par Pydantic avant d'arriver
    ici : une masse hors plage ne touche jamais le modèle et reçoit un 422
    décrivant le champ fautif.
    """
    modele = _modele()
    return Prediction(
        co2_g_par_km=round(
            predire(modele, vehicule.masse, vehicule.puissance, vehicule.boite), 1
        ),
        boite_connue=vehicule.boite in _boites_connues(modele),
    )
