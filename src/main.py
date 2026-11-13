"""Point d'entrée du projet.

    python -m src.main --entrainer   # entraîne et sauvegarde le modèle (une seule fois)
    python -m src.main               # lance l'interface graphique
"""

from __future__ import annotations

import argparse
import sys

from src.application import Application
from src.exceptions import FichierIntrouvableError


def afficher_resume(metadonnees: dict) -> None:
    """Affiche dans le terminal le résultat de l'entraînement."""
    scores = metadonnees["scores_test"]
    refs = metadonnees["references_f1"]
    print(f"Lignes : {metadonnees['lignes_brutes']} brutes, {metadonnees['lignes_propres']} propres")
    print(f"Entraînement : {metadonnees['duree_entrainement_s']} s, "
          f"seuil choisi = {metadonnees['seuil_decision']}")
    print(f"F1 modèle         : {scores['f1']:.3f}  (précision {scores['precision']:.3f}, "
          f"rappel {scores['rappel']:.3f})")
    print(f"F1 réf. majorité  : {refs['majoritaire']:.3f}")
    print(f"F1 réf. top 3     : {refs['depart_top3']:.3f}")


def main(arguments: list[str] | None = None) -> int:
    """Lit les arguments, puis entraîne le modèle ou lance l'interface. Renvoie le code de sortie."""
    analyseur = argparse.ArgumentParser(description="Prédicteur de podium F1 (INF1007).")
    analyseur.add_argument("--entrainer", action="store_true",
                           help="entraîner et sauvegarder le modèle")
    options = analyseur.parse_args(arguments)

    try:
        if options.entrainer:
            from src.entrainement import entrainer_et_sauvegarder
            afficher_resume(entrainer_et_sauvegarder())
            return 0
        application = Application()
    except FichierIntrouvableError as erreur:
        print(f"Erreur : {erreur}")
        print("Si le modèle manque, lancez d'abord : python -m src.main --entrainer")
        return 1

    from src.interface import FenetrePrincipale  # importé ici : Tkinter seulement si besoin
    FenetrePrincipale(application).lancer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
