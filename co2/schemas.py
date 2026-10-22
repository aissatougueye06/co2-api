"""Forme des requêtes et des réponses de l'API.

C'est ici qu'on valide. Le modèle, lui, n'a aucune notion de plausibilité :
une masse négative lui donne un CO₂ négatif sans broncher. Une API exposée
reçoit n'importe quoi, donc on filtre à la frontière — une fois, au point
d'entrée, pas dans la fonction de prédiction qui est une bibliothèque.
"""

from pydantic import BaseModel, Field


class Vehicule(BaseModel):
    """Un véhicule à évaluer.

    Les bornes couvrent largement le parc homologué : la plus légère du fichier
    ADEME est autour de 825 kg, la plus puissante à 545 kW. On accepte au-delà
    pour ne pas rejeter un véhicule réel, mais pas l'absurde.
    """

    masse: float = Field(
        gt=500,
        lt=3500,
        description="Masse en ordre de marche minimale, en kg",
        examples=[1300],
    )
    puissance: float = Field(
        gt=0,
        lt=1000,
        description="Puissance maximale, en kW",
        examples=[90],
    )
    boite: str = Field(
        min_length=1,
        max_length=10,
        description="Type de boîte et nombre de rapports, format ADEME (ex. 'M 6')",
        examples=["M 6"],
    )


class Prediction(BaseModel):
    """Le CO₂ estimé, et ce qu'il faut savoir pour s'y fier."""

    co2_g_par_km: float = Field(description="Émissions estimées, en g/km")
    boite_connue: bool = Field(
        description=(
            "False si ce type de boîte était absent de l'entraînement : la "
            "prédiction est alors dégradée, calculée comme pour la modalité "
            "de référence."
        )
    )


class Sante(BaseModel):
    """Réponse de la sonde de disponibilité."""

    statut: str
    modele_charge: bool
