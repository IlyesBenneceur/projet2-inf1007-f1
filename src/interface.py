"""Interface graphique Tkinter. Elle affiche et transmet ; la logique est dans Application."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.application import Application
from src.exceptions import DonneeInvalideError
from src.graphiques import figure_courbe_probabilite
from src.variables import VariableCategorielle, VariableNumerique

AVERTISSEMENT = ("Exercice pédagogique (INF1007) : cet outil n'est pas une aide à la décision "
                 "et ne doit pas servir à parier. Le modèle se trompe souvent.")


class FenetrePrincipale:
    """Fenêtre principale : formulaire à gauche, résultat et graphique à droite."""

    def __init__(self, application: Application) -> None:
        self._application = application
        self._racine = tk.Tk()
        self._racine.title("Prédicteur de podium F1")
        self._champs: dict[str, tk.StringVar] = {}
        self._canevas: FigureCanvasTkAgg | None = None

        formulaire = ttk.Frame(self._racine, padding=12)
        formulaire.grid(row=0, column=0, sticky="ns")
        droite = ttk.Frame(self._racine, padding=12)
        droite.grid(row=0, column=1, sticky="nsew")
        self._racine.columnconfigure(1, weight=1)
        self._racine.rowconfigure(0, weight=1)

        self._construire_formulaire(formulaire)
        self._construire_resultat(droite)

    def _construire_formulaire(self, cadre: ttk.Frame) -> None:
        """Un champ par variable, choisi selon son type (construit depuis les métadonnées)."""
        ttk.Label(cadre, text="Caractéristiques du pilote", font=("", 13, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        for ligne, variable in enumerate(self._application.variables, start=1):
            valeur = tk.StringVar()
            self._champs[variable.nom] = valeur
            if isinstance(variable, VariableNumerique):
                texte = f"{variable.libelle} ({variable.minimum:g} à {variable.maximum:g})"
                champ = ttk.Entry(cadre, textvariable=valeur, width=24)
            else:
                texte = variable.libelle
                champ = ttk.Combobox(cadre, textvariable=valeur, width=22, state="readonly",
                                     values=variable.valeurs if isinstance(
                                         variable, VariableCategorielle) else [])
            ttk.Label(cadre, text=texte).grid(row=ligne, column=0, sticky="w", pady=3)
            champ.grid(row=ligne, column=1, sticky="ew", pady=3, padx=(8, 0))

        suivant = len(self._champs) + 1
        ttk.Button(cadre, text="Prédire", command=self._sur_prediction).grid(
            row=suivant, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        ttk.Button(cadre, text="Voir l'historique", command=self._afficher_historique).grid(
            row=suivant + 1, column=0, columnspan=2, sticky="ew")

        meta = self._application.metadonnees
        resume = (f"Modèle : {meta['modele']}\n"
                  f"F1 sur le test : {meta['scores_test']['f1']:.2f} "
                  f"(règle « départ top 3 » : {meta['references_f1']['depart_top3']:.2f})")
        ttk.Label(cadre, text=resume, foreground="gray").grid(
            row=suivant + 2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(cadre, text=AVERTISSEMENT, foreground="#b03030", wraplength=330).grid(
            row=suivant + 3, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _construire_resultat(self, cadre: ttk.Frame) -> None:
        """Zone de résultat texte et emplacement du graphique."""
        self._resultat = ttk.Label(cadre, text="Remplissez le formulaire puis cliquez sur « Prédire ».",
                                   font=("", 12), justify="left")
        self._resultat.pack(anchor="w", pady=(0, 8))
        self._cadre_graphique = ttk.Frame(cadre)
        self._cadre_graphique.pack(fill="both", expand=True)
        figure = Figure(figsize=(6, 4))
        axe = figure.add_subplot()
        axe.text(0.5, 0.5, "Le graphique apparaîtra après la première prédiction.",
                 ha="center", va="center")
        axe.set_axis_off()
        self._afficher_figure(figure)

    def _afficher_figure(self, figure: Figure) -> None:
        """Remplace le graphique affiché par une nouvelle figure."""
        if self._canevas is not None:
            self._canevas.get_tk_widget().destroy()
        self._canevas = FigureCanvasTkAgg(figure, master=self._cadre_graphique)
        self._canevas.draw()
        self._canevas.get_tk_widget().pack(fill="both", expand=True)

    def _lire_saisies(self) -> dict[str, str]:
        return {nom: valeur.get() for nom, valeur in self._champs.items()}

    def _sur_prediction(self) -> None:
        """Bouton « Prédire » : une saisie invalide affiche un message, jamais un plantage."""
        saisies = self._lire_saisies()
        try:
            resultat = self._application.predire(saisies)
            positions, probabilites = self._application.courbe_selon_position(saisies)
        except DonneeInvalideError as erreur:
            messagebox.showerror("Saisie invalide", str(erreur), parent=self._racine)
            return
        couleur = "#1a7f37" if resultat.podium_predit else "#444444"
        self._resultat.configure(text=str(resultat), foreground=couleur)
        position = int(saisies["position_depart"].strip())
        self._afficher_figure(figure_courbe_probabilite(
            positions, probabilites, position, resultat.probabilite, resultat.seuil))

    def _afficher_historique(self) -> None:
        """Ouvre une fenêtre avec le tableau des prédictions passées."""
        lignes = self._application.historique()
        fenetre = tk.Toplevel(self._racine)
        fenetre.title(f"Historique des prédictions ({len(lignes)})")
        if not lignes:
            ttk.Label(fenetre, text="Aucune prédiction pour l'instant.", padding=20).pack()
            return
        colonnes = list(lignes[0].keys())
        tableau = ttk.Treeview(fenetre, columns=colonnes, show="headings", height=15)
        for colonne in colonnes:
            tableau.heading(colonne, text=colonne)
            tableau.column(colonne, width=110, anchor="center")
        for ligne in lignes:
            tableau.insert("", "end", values=[ligne[c] for c in colonnes])
        tableau.pack(fill="both", expand=True, padx=8, pady=8)

    def lancer(self) -> None:
        """Démarre la boucle d'événements Tkinter."""
        self._racine.mainloop()
