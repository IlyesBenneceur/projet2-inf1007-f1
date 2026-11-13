"""Chaîne complète d'entraînement : données → modèle → fichiers sauvegardés.

Lancée une seule fois avec `python -m src.main --entrainer` ; l'interface
recharge ensuite le modèle sans le ré-entraîner.
"""

from __future__ import annotations

import os
from datetime import date

from src.application import DOSSIER_MODELES, NOM_METADONNEES, NOM_MODELE
from src.donnees import (CHEMIN_DONNEES, COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES,
                         DOSSIER_RACINE, JeuDeDonnees)
from src.graphiques import figure_matrice_confusion, figure_taux_podium_par_depart
from src.modele import (ModelePodium, ecrire_metadonnees, f1_numpy, predire_reference_grille,
                        predire_reference_majoritaire)
from src.variables import VariableCategorielle, VariableNumerique

DOSSIER_GRAPHIQUES = os.path.join(DOSSIER_RACINE, "graphiques")
GRAINE = 42

LIBELLES = {
    "position_depart": "Position de départ sur la grille",
    "annee": "Année de la course",
    "ecurie": "Écurie",
    "motoriste": "Motoriste",
    "manufacturier_pneus": "Manufacturier de pneus",
    "circuit": "Circuit",
    "type_circuit": "Type de circuit",
}


def construire_variables(jeu: JeuDeDonnees) -> list:
    """Crée une Variable par caractéristique à partir des données propres."""
    numeriques = [VariableNumerique(nom, LIBELLES[nom], mini, maxi, entier=True)
                  for nom, (mini, maxi) in jeu.intervalles_numeriques().items()]
    categorielles = [VariableCategorielle(nom, LIBELLES[nom], valeurs)
                     for nom, valeurs in jeu.valeurs_categorielles().items()]
    return numeriques + categorielles


def entrainer_et_sauvegarder(chemin_donnees: str = CHEMIN_DONNEES,
                             dossier_modeles: str = DOSSIER_MODELES,
                             dossier_graphiques: str = DOSSIER_GRAPHIQUES) -> dict:
    """Charge, nettoie, entraîne, évalue et sauvegarde le modèle et ses graphiques.

    Renvoie le dictionnaire des métadonnées écrit dans le JSON.
    Lève FichierIntrouvableError si le CSV est absent.
    """
    jeu = JeuDeDonnees(chemin_donnees)
    jeu.charger()
    figure_taux_podium_par_depart(jeu.donnees).savefig(
        os.path.join(dossier_graphiques, "exploration_taux_podium.png"), dpi=120)

    x_train, x_test, y_train, y_test = jeu.separer(taille_test=0.2, graine=GRAINE)

    references = {
        "majoritaire": f1_numpy(y_test, predire_reference_majoritaire(y_train, len(y_test))),
        "depart_top3": f1_numpy(y_test, predire_reference_grille(x_test["position_depart"])),
    }

    modele = ModelePodium(COLONNES_NUMERIQUES, COLONNES_CATEGORIELLES, graine=GRAINE)
    modele.entrainer(x_train, y_train)
    scores = modele.evaluer(x_test, y_test)
    figure_matrice_confusion(y_test, modele.predire(x_test)).savefig(
        os.path.join(dossier_graphiques, "matrice_confusion.png"), dpi=120)

    modele.sauvegarder(os.path.join(dossier_modeles, NOM_MODELE))
    metadonnees = {
        "date_entrainement": date.today().isoformat(),
        "modele": "Régression logistique (scikit-learn)",
        "cible": "podium (1 = termine dans les 3 premiers)",
        "lignes_brutes": jeu.nb_lignes_brutes,
        "lignes_propres": len(jeu),
        "lignes_entrainement": len(x_train),
        "lignes_test": len(x_test),
        "graine": GRAINE,
        "taux_podium": float(jeu.cible().mean()),
        "seuil_decision": modele.seuil,
        "duree_entrainement_s": round(ModelePodium.entrainer.derniere_duree, 2),
        "metrique": "F1 (classe podium)",
        "scores_test": scores,
        "references_f1": references,
        "variables": [v.vers_metadonnees() for v in construire_variables(jeu)],
    }
    ecrire_metadonnees(metadonnees, os.path.join(dossier_modeles, NOM_METADONNEES))
    return metadonnees
