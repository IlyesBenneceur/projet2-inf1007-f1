"""Extrait de F1DB la copie du jeu de données publiée dans donnees/.

Usage (à lancer une seule fois, depuis la racine du dépôt) :
    python outils/extraire_f1db.py chemin/vers/f1db-csv

Le dossier f1db-csv provient de l'archive f1db-csv.zip publiée sur
https://github.com/f1db/f1db/releases (licence CC BY 4.0).

On garde uniquement les colonnes utiles au projet. Les colonnes qui
identifient les pilotes (driverId, driverNumber) sont retirées : elles ne
servent pas au modèle et le dépôt est public.
"""

import os
import sys

import pandas as pd

COLONNES_RESULTATS = [
    "raceId", "year", "round", "positionText", "positionNumber",
    "gridPositionText", "gridPositionNumber", "constructorId",
    "engineManufacturerId", "tyreManufacturerId", "sharedCar",
]
COLONNES_COURSES = ["id", "circuitId", "circuitType"]


def extraire(dossier_f1db: str, chemin_sortie: str) -> int:
    """Joint les résultats de course et les circuits, puis écrit le CSV réduit.

    Reçoit le dossier des CSV F1DB et le chemin du fichier à écrire.
    Renvoie le nombre de lignes écrites.
    Lève FileNotFoundError si un des deux fichiers F1DB est absent.
    """
    chemin_resultats = os.path.join(dossier_f1db, "f1db-races-race-results.csv")
    chemin_courses = os.path.join(dossier_f1db, "f1db-races.csv")
    for chemin in (chemin_resultats, chemin_courses):
        if not os.path.exists(chemin):
            raise FileNotFoundError(chemin)

    resultats = pd.read_csv(chemin_resultats, usecols=COLONNES_RESULTATS)
    courses = pd.read_csv(chemin_courses, usecols=COLONNES_COURSES)
    courses = courses.rename(columns={"id": "raceId"})

    fusion = resultats.merge(courses, on="raceId", how="left")
    fusion.to_csv(chemin_sortie, index=False)
    return len(fusion)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sortie = os.path.join(racine, "donnees", "f1_resultats.csv")
    print(f"{extraire(sys.argv[1], sortie)} lignes écrites dans {sortie}")
