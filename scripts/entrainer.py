"""Entraîne le modèle à partir du CSV ADEME et l'écrit sur disque.

    python scripts/entrainer.py data/vehicules.csv

Ce script est le seul endroit où le modèle est entraîné. L'API se contente de
charger le fichier qu'il produit : entraîner au démarrage d'un service rendrait
chaque redéploiement imprévisible.
"""

import argparse
from pathlib import Path

from co2.modele import entrainer, sauvegarder
from co2.preparation import charger, preparer


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("csv", type=Path, help="chemin du fichier ADEME")
    parseur.add_argument("--energ", default="ES", help="motorisation (GO ou ES)")
    parseur.add_argument("--sortie", type=Path, default=Path("models/co2.joblib"))
    args = parseur.parse_args()

    donnees = preparer(charger(args.csv), args.energ)
    print(f"{len(donnees)} vehicules {args.energ} apres preparation")

    modele, metriques = entrainer(donnees)
    print(metriques)

    chemin = sauvegarder(modele, args.sortie)
    print(f"Modele ecrit dans {chemin}")


if __name__ == "__main__":
    main()
