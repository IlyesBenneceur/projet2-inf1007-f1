"""Variables d'entrée de l'application : chacune sait valider la saisie de l'utilisateur."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.exceptions import DonneeInvalideError


class Variable(ABC):
    """Classe parente abstraite d'une variable d'entrée du modèle."""

    def __init__(self, nom: str, libelle: str) -> None:
        """Reçoit le nom de colonne (pour le modèle) et le libellé (pour l'utilisateur)."""
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne dans le modèle."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Libellé affiché à l'utilisateur."""
        return self._libelle

    def _verifier_non_vide(self, saisie: str) -> str:
        """Renvoie la saisie sans espaces autour. Lève DonneeInvalideError si elle est vide."""
        texte = saisie.strip()
        if not texte:
            raise DonneeInvalideError(f"« {self._libelle} » est vide.")
        return texte

    @abstractmethod
    def valider(self, saisie: str) -> int | float | str:
        """Convertit la saisie texte en valeur utilisable par le modèle.

        Lève DonneeInvalideError avec un message clair si la saisie est invalide.
        """

    @abstractmethod
    def vers_metadonnees(self) -> dict:
        """Renvoie la description de la variable pour le fichier JSON de métadonnées."""

    def __eq__(self, autre: object) -> bool:
        if not isinstance(autre, Variable):
            return NotImplemented
        return self.vers_metadonnees() == autre.vers_metadonnees()

    def __str__(self) -> str:
        return self._libelle


class VariableNumerique(Variable):
    """Variable numérique bornée par un intervalle [minimum, maximum]."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float,
                 entier: bool = False) -> None:
        """Lève DonneeInvalideError si minimum > maximum."""
        super().__init__(nom, libelle)
        if minimum > maximum:
            raise DonneeInvalideError(f"Intervalle invalide pour {nom} : {minimum} > {maximum}.")
        self._minimum = minimum
        self._maximum = maximum
        self._entier = entier

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    def valider(self, saisie: str) -> int | float:
        """Accepte un nombre (virgule ou point) dans l'intervalle ; entier si demandé."""
        texte = self._verifier_non_vide(saisie).replace(",", ".")
        try:
            valeur = float(texte)
        except ValueError:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être un nombre (reçu : « {saisie.strip()} »)."
            ) from None
        if self._entier and not valeur.is_integer():
            raise DonneeInvalideError(f"« {self._libelle} » doit être un nombre entier.")
        if not self._minimum <= valeur <= self._maximum:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être entre {self._format(self._minimum)} "
                f"et {self._format(self._maximum)}."
            )
        return int(valeur) if self._entier else valeur

    def _format(self, nombre: float) -> str:
        """Affiche 3 plutôt que 3.0 pour une variable entière."""
        return str(int(nombre)) if self._entier else str(nombre)

    def vers_metadonnees(self) -> dict:
        return {"nom": self._nom, "libelle": self._libelle, "type": "numerique",
                "minimum": self._minimum, "maximum": self._maximum, "entier": self._entier}

    def __repr__(self) -> str:
        return f"VariableNumerique({self._nom!r}, [{self._minimum}, {self._maximum}])"


class VariableCategorielle(Variable):
    """Variable dont la valeur doit faire partie d'une liste de choix."""

    def __init__(self, nom: str, libelle: str, valeurs: list[str]) -> None:
        """Lève DonneeInvalideError si la liste de valeurs est vide."""
        super().__init__(nom, libelle)
        if not valeurs:
            raise DonneeInvalideError(f"Aucune valeur possible pour {nom}.")
        # Tri alphabétique sans tenir compte des majuscules (clé de tri lambda).
        self._valeurs = sorted(set(valeurs), key=lambda v: v.lower())

    @property
    def valeurs(self) -> list[str]:
        """Copie de la liste des choix (la liste interne reste protégée)."""
        return list(self._valeurs)

    def valider(self, saisie: str) -> str:
        """Accepte uniquement une des valeurs connues du modèle."""
        texte = self._verifier_non_vide(saisie)
        if texte not in self._valeurs:
            raise DonneeInvalideError(
                f"« {texte} » n'est pas une valeur connue pour « {self._libelle} »."
            )
        return texte

    def vers_metadonnees(self) -> dict:
        return {"nom": self._nom, "libelle": self._libelle, "type": "categorielle",
                "valeurs": self._valeurs}

    def __repr__(self) -> str:
        return f"VariableCategorielle({self._nom!r}, {len(self._valeurs)} valeurs)"


def variable_depuis_metadonnees(description: dict) -> Variable:
    """Construit la bonne sous-classe de Variable à partir de sa description JSON.

    Reçoit un dictionnaire produit par vers_metadonnees(). Renvoie une Variable.
    Lève DonneeInvalideError si le type est inconnu ou si une clé manque.
    """
    try:
        if description["type"] == "numerique":
            return VariableNumerique(description["nom"], description["libelle"],
                                     description["minimum"], description["maximum"],
                                     description.get("entier", False))
        if description["type"] == "categorielle":
            return VariableCategorielle(description["nom"], description["libelle"],
                                        description["valeurs"])
    except KeyError as erreur:
        raise DonneeInvalideError(f"Métadonnées incomplètes : clé {erreur} manquante.") from None
    raise DonneeInvalideError(f"Type de variable inconnu : {description['type']!r}.")
