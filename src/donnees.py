"""Lecture et nettoyage du jeu de données F1DB (résultats de course de F1)."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import DonneeInvalideError, FichierIntrouvableError

DOSSIER_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_DONNEES = os.path.join(DOSSIER_RACINE, "donnees", "f1_resultats.csv")

# Colonnes du CSV publié (noms d'origine de F1DB).
COLONNES_REQUISES = [
    "raceId", "year", "positionText", "positionNumber", "gridPositionText",
    "gridPositionNumber", "constructorId", "engineManufacturerId",
    "tyreManufacturerId", "sharedCar", "circuitId", "circuitType",
]

# Noms français utilisés dans le reste du projet.
RENOMMAGE = {
    "year": "annee",
    "gridPositionNumber": "position_depart",
    "constructorId": "ecurie",
    "engineManufacturerId": "motoriste",
    "tyreManufacturerId": "manufacturier_pneus",
    "circuitId": "circuit",
    "circuitType": "type_circuit",
}

# Caractéristiques connues AVANT le départ de la course. Les colonnes connues
# seulement après (points, tours, écart, abandon…) sont volontairement exclues.
COLONNES_NUMERIQUES = ["position_depart", "annee"]
COLONNES_CATEGORIELLES = ["ecurie", "motoriste", "manufacturier_pneus", "circuit", "type_circuit"]
CIBLE = "podium"

# Codes F1DB des pilotes qui n'ont pas pris le départ de la course.
NON_PARTANTS = {"DNQ", "DNPQ", "DNS", "DNP", "EX"}


def verifier_fichier(chemin: str) -> None:
    """Vérifie qu'un fichier existe.

    Reçoit un chemin. Ne renvoie rien.
    Lève FichierIntrouvableError si le fichier n'existe pas.
    """
    if not os.path.isfile(chemin):
        raise FichierIntrouvableError(f"Fichier introuvable : {chemin}")


def lire_par_blocs(chemin: str, taille_bloc: int = 5000) -> Iterator[pd.DataFrame]:
    """Générateur qui lit un CSV par blocs de `taille_bloc` lignes.

    Reçoit le chemin du CSV et la taille des blocs (entier > 0).
    Produit (yield) un DataFrame par bloc, sans charger tout le fichier d'un coup.
    Lève FichierIntrouvableError si le fichier n'existe pas et
    DonneeInvalideError si la taille de bloc n'est pas positive.
    """
    if taille_bloc <= 0:
        raise DonneeInvalideError("La taille de bloc doit être positive.")
    verifier_fichier(chemin)
    with pd.read_csv(chemin, chunksize=taille_bloc) as lecteur:
        for bloc in lecteur:
            yield bloc


def charger_csv(chemin: str = CHEMIN_DONNEES, taille_bloc: int = 5000) -> pd.DataFrame:
    """Charge le CSV complet en assemblant les blocs du générateur.

    Reçoit le chemin du CSV. Renvoie un DataFrame.
    Lève FichierIntrouvableError si le fichier n'existe pas et
    DonneeInvalideError s'il est vide ou s'il manque une colonne requise.
    """
    blocs = list(lire_par_blocs(chemin, taille_bloc))
    if not blocs:
        raise DonneeInvalideError(f"Le fichier {chemin} ne contient aucune ligne.")
    donnees = pd.concat(blocs, ignore_index=True)
    manquantes = [col for col in COLONNES_REQUISES if col not in donnees.columns]
    if manquantes:
        raise DonneeInvalideError(f"Colonnes manquantes : {', '.join(manquantes)}")
    return donnees


def retirer_non_partants(donnees: pd.DataFrame) -> pd.DataFrame:
    """Retire les pilotes qui n'ont pas pris le départ et les voitures partagées.

    Reçoit le DataFrame brut. Renvoie une copie filtrée.
    Un pilote non qualifié (DNQ) ou absent au départ (DNS) n'a pas couru :
    le garder fausserait la cible. Les voitures partagées par deux pilotes
    (années 1950) produisent des lignes en double pour un même résultat.
    """
    a_pris_le_depart = ~donnees["positionText"].isin(NON_PARTANTS)
    voiture_seule = ~donnees["sharedCar"].astype(bool)
    return donnees[a_pris_le_depart & voiture_seule].copy()


def imputer_depart_voie_des_stands(donnees: pd.DataFrame) -> pd.DataFrame:
    """Donne une position de départ aux pilotes partis de la voie des stands.

    Reçoit un DataFrame de partants. Renvoie une copie où la position de
    départ des lignes « PL » (pit lane) vaut la dernière place de la grille
    de cette course + 1 : ces pilotes partent bien derrière tout le monde.
    """
    resultat = donnees.copy()
    derniere_place = resultat.groupby("raceId")["gridPositionNumber"].transform("max")
    voie_des_stands = resultat["gridPositionText"] == "PL"
    resultat.loc[voie_des_stands, "gridPositionNumber"] = derniere_place[voie_des_stands] + 1
    return resultat


def creer_cible(donnees: pd.DataFrame) -> pd.Series:
    """Calcule la cible : 1 si le pilote termine dans les 3 premiers, 0 sinon.

    Reçoit un DataFrame avec la colonne positionNumber (vide pour un abandon).
    Renvoie une Series d'entiers 0/1. Un abandon ou une disqualification
    (positionNumber vide) compte comme 0.
    """
    return (donnees["positionNumber"] <= 3).astype(int)


def nettoyer(donnees: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les étapes de nettoyage au DataFrame brut.

    Reçoit le DataFrame brut de F1DB. Renvoie un DataFrame propre qui ne
    contient que les caractéristiques (noms français) et la cible.
    Lève DonneeInvalideError s'il ne reste aucune ligne après nettoyage.
    """
    partants = imputer_depart_voie_des_stands(retirer_non_partants(donnees))
    # Quelques courses anciennes n'ont pas de grille connue : on les retire.
    partants = partants.dropna(subset=["gridPositionNumber"])
    partants[CIBLE] = creer_cible(partants)
    propre = partants.rename(columns=RENOMMAGE)
    propre = propre[COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES + [CIBLE]]
    propre["position_depart"] = propre["position_depart"].astype(int)
    if propre.empty:
        raise DonneeInvalideError("Aucune ligne valide après le nettoyage.")
    return propre.reset_index(drop=True)


class JeuDeDonnees:
    """Jeu de données F1 : chargement, nettoyage et séparation entraînement/test."""

    def __init__(self, chemin: str = CHEMIN_DONNEES) -> None:
        """Mémorise le chemin du CSV ; rien n'est lu tant qu'on n'appelle pas charger()."""
        self._chemin = chemin
        self._brut: pd.DataFrame | None = None
        self._propre: pd.DataFrame | None = None

    def charger(self) -> None:
        """Lit le CSV puis le nettoie. Lève les erreurs de charger_csv et nettoyer."""
        self._brut = charger_csv(self._chemin)
        self._propre = nettoyer(self._brut)

    @property
    def donnees(self) -> pd.DataFrame:
        """Renvoie le DataFrame propre. Lève DonneeInvalideError s'il n'est pas chargé."""
        if self._propre is None:
            raise DonneeInvalideError("Le jeu de données n'a pas encore été chargé.")
        return self._propre

    @property
    def nb_lignes_brutes(self) -> int:
        """Renvoie le nombre de lignes avant nettoyage (0 si rien n'est chargé)."""
        return 0 if self._brut is None else len(self._brut)

    def caracteristiques(self) -> pd.DataFrame:
        """Renvoie les colonnes d'entrée du modèle."""
        return self.donnees[COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES]

    def cible(self) -> pd.Series:
        """Renvoie la colonne cible (podium 0/1)."""
        return self.donnees[CIBLE]

    def separer(self, taille_test: float = 0.2, graine: int = 42) -> tuple:
        """Sépare en ensembles d'entraînement et de test, de façon reproductible.

        Reçoit la proportion du test et la graine aléatoire.
        Renvoie (X_entrainement, X_test, y_entrainement, y_test).
        La séparation est stratifiée : la proportion de podiums est la même
        dans les deux ensembles.
        """
        return train_test_split(
            self.caracteristiques(), self.cible(),
            test_size=taille_test, random_state=graine, stratify=self.cible(),
        )

    def intervalles_numeriques(self) -> dict[str, tuple[float, float]]:
        """Renvoie {colonne: (minimum, maximum)} pour chaque variable numérique."""
        return {
            col: (float(self.donnees[col].min()), float(self.donnees[col].max()))
            for col in COLONNES_NUMERIQUES
        }

    def valeurs_categorielles(self) -> dict[str, list[str]]:
        """Renvoie {colonne: valeurs possibles triées} pour chaque variable catégorielle."""
        return {
            col: sorted(self.donnees[col].astype(str).unique(), key=lambda v: v.lower())
            for col in COLONNES_CATEGORIELLES
        }

    def __len__(self) -> int:
        """Nombre de lignes propres (0 si rien n'est chargé)."""
        return 0 if self._propre is None else len(self._propre)

    def __repr__(self) -> str:
        return f"JeuDeDonnees(chemin={self._chemin!r}, lignes={len(self)})"
