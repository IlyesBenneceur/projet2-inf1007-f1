"""Modèle de prédiction du podium : entraînement, évaluation, sauvegarde."""

from __future__ import annotations

import functools
import json
import os
import time
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError


def chronometrer(fonction: Callable) -> Callable:
    """Décorateur qui mesure la durée d'exécution d'une fonction.

    La durée (en secondes) de la dernière exécution est conservée dans
    l'attribut `derniere_duree` de la fonction décorée ; le résultat de la
    fonction est renvoyé tel quel.
    """
    @functools.wraps(fonction)
    def enveloppe(*args, **kwargs):
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        enveloppe.derniere_duree = time.perf_counter() - debut
        return resultat

    enveloppe.derniere_duree = 0.0
    return enveloppe


def f1_numpy(y_vrai: np.ndarray, y_predit: np.ndarray) -> float:
    """Calcule le score F1 de la classe 1, de façon vectorisée (sans boucle).

    Reçoit deux tableaux de 0/1 de même taille. Renvoie un flottant entre 0 et 1.
    F1 = 2·VP / (2·VP + FP + FN). Renvoie 0.0 s'il n'y a aucun positif réel ni prédit.
    Lève DonneeInvalideError si les tableaux n'ont pas la même taille.
    """
    vrai = np.asarray(y_vrai)
    predit = np.asarray(y_predit)
    if vrai.shape != predit.shape:
        raise DonneeInvalideError(f"Tailles différentes : {vrai.shape} et {predit.shape}.")
    vrais_positifs = np.sum((vrai == 1) & (predit == 1))
    faux_positifs = np.sum((vrai == 0) & (predit == 1))
    faux_negatifs = np.sum((vrai == 1) & (predit == 0))
    denominateur = 2 * vrais_positifs + faux_positifs + faux_negatifs
    return 0.0 if denominateur == 0 else float(2 * vrais_positifs / denominateur)


def predire_reference_majoritaire(y_entrainement: pd.Series, nb_predictions: int) -> np.ndarray:
    """Référence triviale : prédit toujours la classe la plus fréquente à l'entraînement."""
    classe_majoritaire = int(pd.Series(y_entrainement).mode()[0])
    return np.full(nb_predictions, classe_majoritaire)


def predire_reference_grille(positions_depart: pd.Series, seuil: int = 3) -> np.ndarray:
    """Référence « bon sens » : podium si le pilote part dans les `seuil` premiers."""
    return (np.asarray(positions_depart) <= seuil).astype(int)


def choisir_seuil(y_vrai: np.ndarray, probabilites: np.ndarray,
                  candidats: np.ndarray | None = None) -> float:
    """Choisit le seuil de décision qui maximise le F1.

    Reçoit les vraies classes et les probabilités prédites (sur des données
    qui n'ont PAS servi à ajuster le modèle). Renvoie le meilleur seuil.
    """
    if candidats is None:
        candidats = np.round(np.arange(0.10, 0.90, 0.05), 2)
    scores = [(seuil, f1_numpy(y_vrai, (probabilites >= seuil).astype(int))) for seuil in candidats]
    meilleur_seuil, _ = max(scores, key=lambda paire: paire[1])
    return float(meilleur_seuil)


class ModelePodium:
    """Régression logistique qui estime la probabilité de podium d'un pilote."""

    def __init__(self, colonnes_numeriques: list[str], colonnes_categorielles: list[str],
                 graine: int = 42) -> None:
        """Prépare le pipeline (mise à l'échelle + encodage one-hot + régression)."""
        self._colonnes_numeriques = list(colonnes_numeriques)
        self._colonnes_categorielles = list(colonnes_categorielles)
        self._graine = graine
        self._seuil = 0.5
        self._entraine = False
        pretraitement = ColumnTransformer([
            ("numeriques", StandardScaler(), self._colonnes_numeriques),
            ("categorielles", OneHotEncoder(handle_unknown="ignore"), self._colonnes_categorielles),
        ])
        self._pipeline = Pipeline([
            ("pretraitement", pretraitement),
            ("regression", LogisticRegression(max_iter=2000)),
        ])

    @property
    def colonnes(self) -> list[str]:
        """Colonnes attendues, dans l'ordre."""
        return self._colonnes_numeriques + self._colonnes_categorielles

    @property
    def seuil(self) -> float:
        """Seuil de probabilité à partir duquel on prédit un podium."""
        return self._seuil

    @property
    def est_entraine(self) -> bool:
        return self._entraine

    @chronometrer
    def entrainer(self, x_entrainement: pd.DataFrame, y_entrainement: pd.Series) -> None:
        """Choisit le seuil par validation croisée, puis entraîne le modèle final.

        La validation croisée (5 plis) n'utilise que l'ensemble d'entraînement :
        l'ensemble de test reste intact pour l'évaluation finale.
        """
        probabilites_cv = cross_val_predict(
            self._pipeline, x_entrainement, y_entrainement, cv=5, method="predict_proba"
        )[:, 1]
        self._seuil = choisir_seuil(np.asarray(y_entrainement), probabilites_cv)
        self._pipeline.fit(x_entrainement, y_entrainement)
        self._entraine = True

    def _verifier_entraine(self) -> None:
        if not self._entraine:
            raise ModeleNonEntraineError("Le modèle doit être entraîné avant de prédire.")

    def probabilites(self, x: pd.DataFrame) -> np.ndarray:
        """Renvoie la probabilité de podium de chaque ligne de x."""
        self._verifier_entraine()
        return self._pipeline.predict_proba(x[self.colonnes])[:, 1]

    def predire(self, x: pd.DataFrame) -> np.ndarray:
        """Renvoie 1 (podium) ou 0 pour chaque ligne, selon le seuil choisi."""
        return (self.probabilites(x) >= self._seuil).astype(int)

    def evaluer(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Calcule F1, précision, rappel et exactitude sur un ensemble de test."""
        vrai = np.asarray(y_test)
        predit = self.predire(x_test)
        vp = np.sum((vrai == 1) & (predit == 1))
        return {
            "f1": f1_numpy(vrai, predit),
            "precision": float(vp / max(np.sum(predit == 1), 1)),
            "rappel": float(vp / max(np.sum(vrai == 1), 1)),
            "exactitude": float(np.mean(vrai == predit)),
        }

    def sauvegarder(self, chemin: str) -> None:
        """Enregistre le modèle entraîné avec joblib."""
        self._verifier_entraine()
        joblib.dump(self, chemin)

    @classmethod
    def charger(cls, chemin: str) -> ModelePodium:
        """Recharge un modèle sauvegardé.

        Lève FichierIntrouvableError si le fichier n'existe pas et
        DonneeInvalideError s'il ne contient pas un ModelePodium.
        """
        if not os.path.isfile(chemin):
            raise FichierIntrouvableError(f"Modèle introuvable : {chemin}")
        modele = joblib.load(chemin)
        if not isinstance(modele, cls):
            raise DonneeInvalideError(f"{chemin} ne contient pas un ModelePodium.")
        return modele

    def __repr__(self) -> str:
        etat = f"seuil={self._seuil}" if self._entraine else "non entraîné"
        return f"ModelePodium({len(self.colonnes)} variables, {etat})"


def ecrire_metadonnees(metadonnees: dict, chemin: str) -> None:
    """Écrit les métadonnées du modèle dans un fichier JSON (UTF-8, indenté)."""
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(metadonnees, fichier, ensure_ascii=False, indent=2)


def lire_metadonnees(chemin: str) -> dict:
    """Lit le fichier JSON de métadonnées.

    Lève FichierIntrouvableError s'il n'existe pas et DonneeInvalideError
    s'il ne contient pas du JSON valide.
    """
    if not os.path.isfile(chemin):
        raise FichierIntrouvableError(f"Métadonnées introuvables : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        try:
            return json.load(fichier)
        except json.JSONDecodeError as erreur:
            raise DonneeInvalideError(f"JSON invalide dans {chemin} : {erreur}") from None
