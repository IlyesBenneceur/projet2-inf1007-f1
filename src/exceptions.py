"""Exceptions personnalisées du projet."""


class ProjetError(Exception):
    """Classe parente de toutes les erreurs du projet."""


class DonneeInvalideError(ProjetError):
    """Levée quand une saisie ou une donnée est invalide (vide, non numérique, hors intervalle)."""


class FichierIntrouvableError(ProjetError):
    """Levée quand un fichier nécessaire n'existe pas."""