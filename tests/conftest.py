"""Données de test partagées : un petit CSV synthétique au format F1DB."""

import numpy as np
import pandas as pd
import pytest

from src.entrainement import entrainer_et_sauvegarder


def fabriquer_brut(nb_courses: int = 30, pilotes_par_course: int = 10) -> pd.DataFrame:
    """Crée des résultats de course bidon où partir devant aide à finir devant."""
    generateur = np.random.default_rng(0)
    lignes = []
    for course in range(1, nb_courses + 1):
        grille = np.arange(1, pilotes_par_course + 1)
        arrivee = np.argsort(grille + generateur.normal(0, 3, pilotes_par_course)) + 1
        for depart, position in zip(grille, arrivee):
            lignes.append({
                "raceId": course, "year": 2000 + course % 10, "round": 1,
                "positionText": str(position), "positionNumber": float(position),
                "gridPositionText": str(depart), "gridPositionNumber": float(depart),
                "constructorId": ["ferrari", "mclaren", "williams"][depart % 3],
                "engineManufacturerId": ["ferrari", "mercedes"][depart % 2],
                "tyreManufacturerId": "pirelli", "sharedCar": False,
                "circuitId": ["monza", "monaco"][course % 2],
                "circuitType": ["RACE", "STREET"][course % 2],
            })
    return pd.DataFrame(lignes)


@pytest.fixture
def brut() -> pd.DataFrame:
    return fabriquer_brut()


@pytest.fixture
def chemin_csv(tmp_path, brut) -> str:
    chemin = tmp_path / "f1.csv"
    brut.to_csv(chemin, index=False)
    return str(chemin)


@pytest.fixture
def dossier_modele_entraine(tmp_path, chemin_csv) -> str:
    """Entraîne un vrai modèle sur les données synthétiques, dans un dossier temporaire."""
    modeles = tmp_path / "modeles"
    graphiques = tmp_path / "graphiques"
    modeles.mkdir()
    graphiques.mkdir()
    entrainer_et_sauvegarder(chemin_csv, str(modeles), str(graphiques))
    return str(modeles)
