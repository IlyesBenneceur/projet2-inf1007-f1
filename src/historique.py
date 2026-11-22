"""Historique des prédictions, enregistré dans un fichier CSV."""

from __future__ import annotations

import csv
import os
from datetime import datetime

from src.exceptions import DonneeInvalideError


class Historique:
    """Ajoute et relit les prédictions dans un fichier CSV."""

    def __init__(self, chemin: str, colonnes_saisies: list[str]) -> None:
        """Reçoit le chemin du CSV et les noms des variables saisies."""
        self._chemin = chemin
        self._colonnes = ["horodatage", *colonnes_saisies, "probabilite", "verdict"]

    def ajouter(self, saisies: dict, probabilite: float, verdict: str) -> dict:
        """Ajoute une prédiction à la fin du fichier (crée le fichier au besoin).

        Renvoie la ligne écrite (dictionnaire).
        """
        ligne = {"horodatage": datetime.now().isoformat(timespec="seconds"), **saisies,
                 "probabilite": round(probabilite, 4), "verdict": verdict}
        fichier_nouveau = not os.path.isfile(self._chemin)
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            ecrivain = csv.DictWriter(fichier, fieldnames=self._colonnes)
            if fichier_nouveau:
                ecrivain.writeheader()
            ecrivain.writerow(ligne)
        return ligne

    def lire(self) -> list[dict]:
        """Renvoie toutes les prédictions, de la plus récente à la plus ancienne.

        Renvoie une liste vide si le fichier n'existe pas encore.
        Lève DonneeInvalideError si le fichier est illisible (par exemple
        réenregistré par Excel dans un autre encodage).
        Les lignes sont ajoutées dans l'ordre chronologique : on inverse simplement
        la liste, ce qui reste juste même pour deux prédictions dans la même seconde.
        """
        if not os.path.isfile(self._chemin):
            return []
        try:
            with open(self._chemin, newline="", encoding="utf-8") as fichier:
                lignes = list(csv.DictReader(fichier))
        except (UnicodeDecodeError, csv.Error) as erreur:
            raise DonneeInvalideError(f"Historique illisible : {erreur}") from None
        return list(reversed(lignes))

    @property
    def colonnes(self) -> list[str]:
        """Copie de la liste des colonnes du fichier CSV."""
        return list(self._colonnes)

    def __len__(self) -> int:
        """Nombre de prédictions enregistrées (relit le fichier)."""
        return len(self.lire())