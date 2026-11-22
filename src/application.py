"""Logique de l'application, indépendante de l'interface graphique.

L'interface Tkinter appelle cette classe ; elle ne contient pas de calcul.
C'est ce qui permet de tester la logique avec pytest.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.donnees import DOSSIER_RACINE
from src.exceptions import DonneeInvalideError
from src.historique import Historique
from src.modele import ModelePodium, lire_metadonnees
from src.variables import Variable, variable_depuis_metadonnees

DOSSIER_MODELES = os.path.join(DOSSIER_RACINE, "modeles")
CHEMIN_HISTORIQUE = os.path.join(DOSSIER_RACINE, "historique", "predictions.csv")
NOM_MODELE = "modele_podium.joblib"
NOM_METADONNEES = "metadonnees.json"
# Clés du JSON utilisées par l'application et par l'interface.
CLES_METADONNEES = ("modele", "taux_podium", "scores_test", "references_f1", "variables")


def verifier_metadonnees(metadonnees: object) -> dict:
    """Vérifie que les métadonnées lues dans le JSON sont utilisables.

    Reçoit l'objet lu dans le JSON. Le renvoie tel quel s'il est valide.
    Lève DonneeInvalideError si ce n'est pas un dictionnaire, s'il manque une clé
    ou si la liste des variables ne contient pas la position de départ.
    """
    if not isinstance(metadonnees, dict):
        raise DonneeInvalideError("Les métadonnées doivent être un objet JSON (dictionnaire).")
    manquantes = [cle for cle in CLES_METADONNEES if cle not in metadonnees]
    if manquantes:
        raise DonneeInvalideError(f"Métadonnées incomplètes : {', '.join(manquantes)} manquant(s).")
    if not isinstance(metadonnees["scores_test"], dict) or "f1" not in metadonnees["scores_test"]:
        raise DonneeInvalideError("Métadonnées incomplètes : score F1 du test manquant.")
    if (not isinstance(metadonnees["references_f1"], dict)
            or "depart_top3" not in metadonnees["references_f1"]):
        raise DonneeInvalideError("Métadonnées incomplètes : référence « départ top 3 » manquante.")
    if not isinstance(metadonnees["variables"], list):
        raise DonneeInvalideError("Métadonnées invalides : « variables » doit être une liste.")
    return metadonnees


class ResultatPrediction:
    """Résultat d'une prédiction, formulé pour un non-informaticien."""

    def __init__(self, probabilite: float, seuil: float, taux_moyen: float) -> None:
        """Reçoit la probabilité de podium (0 à 1), le seuil du modèle et le taux moyen de podium."""
        self.probabilite = probabilite
        self.seuil = seuil
        self.taux_moyen = taux_moyen

    @property
    def podium_predit(self) -> bool:
        """Vrai si la probabilité atteint le seuil de décision."""
        return self.probabilite >= self.seuil

    @property
    def verdict(self) -> str:
        """Renvoie « Podium probable » ou « Podium peu probable »."""
        return "Podium probable" if self.podium_predit else "Podium peu probable"

    def __str__(self) -> str:
        """Texte affiché à l'utilisateur : verdict, probabilité et comparaison à la moyenne."""
        if self.probabilite >= self.taux_moyen:
            comparaison = "plus que"
        else:
            comparaison = "moins que"
        return (f"{self.verdict}\n"
                f"Probabilité estimée de finir dans les 3 premiers : {self.probabilite:.0%}\n"
                f"C'est {comparaison} la moyenne de tous les pilotes ({self.taux_moyen:.0%}).")


class Application:
    """Contient le modèle, les variables d'entrée et l'historique (composition)."""

    def __init__(self, dossier_modeles: str = DOSSIER_MODELES,
                 chemin_historique: str = CHEMIN_HISTORIQUE) -> None:
        """Charge le modèle et ses métadonnées.

        Lève FichierIntrouvableError si le modèle n'a pas encore été entraîné et
        DonneeInvalideError si le modèle ou ses métadonnées sont invalides.
        """
        self._metadonnees = verifier_metadonnees(
            lire_metadonnees(os.path.join(dossier_modeles, NOM_METADONNEES)))
        self._modele = ModelePodium.charger(os.path.join(dossier_modeles, NOM_MODELE))
        self._variables: list[Variable] = [
            variable_depuis_metadonnees(description)
            for description in self._metadonnees["variables"]
        ]
        if "position_depart" not in [v.nom for v in self._variables]:
            raise DonneeInvalideError("Métadonnées incomplètes : variable position_depart absente.")
        self._historique = Historique(chemin_historique, [v.nom for v in self._variables])

    @property
    def variables(self) -> list[Variable]:
        """Copie de la liste des variables d'entrée (la liste interne reste protégée)."""
        return list(self._variables)

    @property
    def metadonnees(self) -> dict:
        """Métadonnées du modèle, lues dans le fichier JSON."""
        return self._metadonnees

    def valider(self, saisies: dict[str, str]) -> dict[str, int | float | str]:
        """Valide chaque saisie avec sa variable (polymorphisme : valider()).

        Reçoit {nom: texte saisi}. Renvoie {nom: valeur convertie}.
        Lève DonneeInvalideError en regroupant tous les messages d'erreur.
        """
        valeurs = {}
        erreurs = []
        for variable in self._variables:
            try:
                valeurs[variable.nom] = variable.valider(saisies.get(variable.nom, ""))
            except DonneeInvalideError as erreur:
                erreurs.append(str(erreur))
        if erreurs:
            raise DonneeInvalideError("\n".join(erreurs))
        return valeurs

    def predire(self, saisies: dict[str, str]) -> ResultatPrediction:
        """Valide les saisies, calcule la probabilité de podium et l'ajoute à l'historique."""
        valeurs = self.valider(saisies)
        probabilite = float(self._modele.probabilites(pd.DataFrame([valeurs]))[0])
        resultat = ResultatPrediction(probabilite, self._modele.seuil,
                                      self._metadonnees["taux_podium"])
        self._historique.ajouter(valeurs, probabilite, resultat.verdict)
        return resultat

    def courbe_selon_position(self, saisies: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        """Probabilité de podium pour chaque position de départ possible.

        Les autres caractéristiques restent celles saisies. Renvoie
        (positions, probabilités). Lève DonneeInvalideError si une saisie est invalide.
        """
        valeurs = self.valider(saisies)
        position = next(v for v in self._variables if v.nom == "position_depart")
        positions = np.arange(int(position.minimum), int(position.maximum) + 1)
        tableau = pd.DataFrame([valeurs] * len(positions))
        tableau["position_depart"] = positions
        return positions, self._modele.probabilites(tableau)

    def historique(self) -> list[dict]:
        """Renvoie les prédictions passées, de la plus récente à la plus ancienne."""
        return self._historique.lire()

    def __repr__(self) -> str:
        """Représentation pour le débogage : modèle et nombre de variables."""
        return f"Application({self._modele!r}, {len(self._variables)} variables)"