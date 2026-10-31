"""Télécharge le fichier ADEME 2015 et vérifie son empreinte.

    python scripts/telecharger_donnees.py

Sans argument, écrit `data/vehicules.csv`. Utilisé en local comme pendant la
construction de l'image Docker : les deux obtiennent exactement le même fichier.
"""

import argparse
import sys
from pathlib import Path

from co2.donnees import (
    CSV_ADEME_2015,
    SHA256_ADEME_2015,
    URL_ADEME_2015,
    EmpreinteInvalide,
    telecharger,
)


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--sortie", type=Path, default=Path("data/vehicules.csv"))
    args = parseur.parse_args()

    try:
        chemin = telecharger(
            URL_ADEME_2015, CSV_ADEME_2015, args.sortie, SHA256_ADEME_2015
        )
    except EmpreinteInvalide as erreur:
        print(erreur, file=sys.stderr)
        return 1
    print(f"Donnees verifiees dans {chemin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
