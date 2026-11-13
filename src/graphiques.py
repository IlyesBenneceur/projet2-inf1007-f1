"""Graphiques Matplotlib du projet.

Les fonctions renvoient des objets Figure sans passer par pyplot : la même
figure peut être enregistrée sur disque (savefig) ou intégrée dans Tkinter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def figure_taux_podium_par_depart(donnees: pd.DataFrame, position_max: int = 20) -> Figure:
    """Graphique d'exploration : % de podiums selon la position de départ.

    Reçoit le DataFrame propre (colonnes position_depart et podium).
    Renvoie une Figure en barres.
    """
    taux = donnees.groupby("position_depart")["podium"].mean().loc[1:position_max] * 100
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axe = figure.add_subplot()
    axe.bar(taux.index, taux.values, color="#1f77b4")
    axe.axhline(donnees["podium"].mean() * 100, color="#d62728", linestyle="--",
                label=f"Moyenne générale ({donnees['podium'].mean():.0%})")
    axe.set_title("Taux de podium selon la position de départ (F1, 1950-2026)")
    axe.set_xlabel("Position de départ sur la grille")
    axe.set_ylabel("Pilotes terminant sur le podium (%)")
    axe.set_xticks(range(1, position_max + 1))
    axe.legend()
    return figure


def figure_matrice_confusion(y_vrai: np.ndarray, y_predit: np.ndarray) -> Figure:
    """Matrice de confusion (2×2) du modèle sur l'ensemble de test."""
    vrai = np.asarray(y_vrai)
    predit = np.asarray(y_predit)
    matrice = np.array([[np.sum((vrai == r) & (predit == p)) for p in (0, 1)] for r in (0, 1)])
    figure = Figure(figsize=(5, 4.5), layout="constrained")
    axe = figure.add_subplot()
    axe.imshow(matrice, cmap="Blues")
    for ligne in range(2):
        for colonne in range(2):
            couleur = "white" if matrice[ligne, colonne] > matrice.max() / 2 else "black"
            axe.text(colonne, ligne, str(matrice[ligne, colonne]), ha="center", va="center",
                     color=couleur, fontsize=14)
    etiquettes = ["Pas de podium", "Podium"]
    axe.set_xticks([0, 1], etiquettes)
    axe.set_yticks([0, 1], etiquettes)
    axe.set_xlabel("Prédit par le modèle")
    axe.set_ylabel("Résultat réel")
    axe.set_title("Matrice de confusion (ensemble de test)")
    return figure


def figure_courbe_probabilite(positions: np.ndarray, probabilites: np.ndarray,
                              position_choisie: int, probabilite_choisie: float,
                              seuil: float) -> Figure:
    """Graphique de l'interface : probabilité de podium selon la position de départ.

    Toutes les autres caractéristiques restent celles saisies par l'utilisateur ;
    le point rouge marque la prédiction demandée.
    """
    figure = Figure(figsize=(6, 4), layout="constrained")
    axe = figure.add_subplot()
    axe.plot(positions, probabilites * 100, marker="o", markersize=3, label="Probabilité estimée")
    axe.axhline(seuil * 100, color="gray", linestyle="--", label=f"Seuil de décision ({seuil:.0%})")
    axe.scatter([position_choisie], [probabilite_choisie * 100], color="#d62728", s=90,
                zorder=3, label="Votre saisie")
    axe.set_title("Si le pilote partait d'une autre position…")
    axe.set_xlabel("Position de départ sur la grille")
    axe.set_ylabel("Probabilité de podium (%)")
    axe.set_ylim(0, 100)
    axe.legend()
    return figure
