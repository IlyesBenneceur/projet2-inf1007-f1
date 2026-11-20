"""Interface graphique Tkinter. Elle affiche et transmet ; la logique est dans Application."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.application import Application, ResultatPrediction
from src.exceptions import DonneeInvalideError
from src.graphiques import figure_courbe_probabilite
from src.variables import VariableCategorielle, VariableNumerique

AVERTISSEMENT = ("⚠  Exercice pédagogique (INF1007) : cet outil n'est pas une aide à la décision "
                 "et ne doit pas servir à parier. Le modèle se trompe souvent.")

# Palette de couleurs de l'interface.
FOND = "#f4f5f7"
ROUGE_F1 = "#c8102e"
TEXTE = "#1f2328"
TEXTE_SECONDAIRE = "#656d76"
VERT = "#1a7f37"
VERT_CLAIR = "#dafbe1"
GRIS = "#57606a"
GRIS_CLAIR = "#eaeef2"
POLICE = "Helvetica"


class FenetrePrincipale:
    """Fenêtre principale : en-tête, formulaire à gauche, résultat et graphique à droite."""

    def __init__(self, application: Application) -> None:
        self._application = application
        self._racine = tk.Tk()
        self._forcer_mode_clair(self._racine)
        self._racine.title("Prédicteur de podium F1")
        self._racine.configure(bg=FOND)
        self._racine.minsize(1050, 640)
        self._champs: dict[str, tk.StringVar] = {}
        self._canevas: FigureCanvasTkAgg | None = None
        self._configurer_styles()

        self._construire_entete()
        corps = tk.Frame(self._racine, bg=FOND)
        corps.pack(fill="both", expand=True, padx=16, pady=(12, 16))
        gauche = tk.Frame(corps, bg=FOND)
        gauche.pack(side="left", fill="y", padx=(0, 16))
        droite = tk.Frame(corps, bg=FOND)
        droite.pack(side="left", fill="both", expand=True)

        self._construire_formulaire(gauche)
        self._construire_carte_resultat(droite)
        self._construire_graphique(droite)

    @staticmethod
    def _forcer_mode_clair(fenetre: tk.Misc) -> None:
        """Sur macOS en mode sombre, force l'apparence claire pour garder des couleurs lisibles."""
        try:
            fenetre.tk.call("::tk::unsupported::MacWindowStyle", "appearance", fenetre, "aqua")
        except tk.TclError:
            pass  # Pas sur macOS (ou Tk trop ancien) : rien à faire.

    def _configurer_styles(self) -> None:
        """Polices des boutons ttk."""
        style = ttk.Style(self._racine)
        style.configure("Principal.TButton", font=(POLICE, 14, "bold"), padding=8)
        style.configure("Secondaire.TButton", font=(POLICE, 12), padding=4)

    def _construire_entete(self) -> None:
        """Bandeau rouge avec le titre de l'application."""
        entete = tk.Frame(self._racine, bg=ROUGE_F1)
        entete.pack(fill="x")
        tk.Label(entete, text="🏁  Prédicteur de podium F1", bg=ROUGE_F1, fg="white",
                 font=(POLICE, 20, "bold")).pack(side="left", padx=16, pady=(10, 0))
        tk.Label(entete, text="Le pilote finira-t-il dans les 3 premiers ?", bg=ROUGE_F1,
                 fg="#ffd7dc", font=(POLICE, 12)).pack(side="left", padx=4, pady=(14, 0))
        tk.Frame(entete, bg=ROUGE_F1, height=10).pack(side="bottom", fill="x")

    def _construire_formulaire(self, cadre: tk.Frame) -> None:
        """Un champ par variable, choisi selon son type (construit depuis les métadonnées)."""
        groupe = tk.Frame(cadre, bg="white", highlightbackground="#d0d7de",
                          highlightthickness=1, padx=14, pady=12)
        groupe.pack(fill="x")
        tk.Label(groupe, text="Caractéristiques du pilote", bg="white", fg=TEXTE,
                 font=(POLICE, 14, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        for ligne, variable in enumerate(self._application.variables, start=1):
            valeur = tk.StringVar()
            self._champs[variable.nom] = valeur
            tk.Label(groupe, text=variable.libelle, bg="white", fg=TEXTE,
                     font=(POLICE, 12, "bold")).grid(row=2 * ligne, column=0, sticky="w",
                                                     pady=(8, 2))
            if isinstance(variable, VariableNumerique):
                champ = ttk.Entry(groupe, textvariable=valeur, width=28, font=(POLICE, 12))
                aide = f"Nombre entier de {variable.minimum:g} à {variable.maximum:g}"
            elif isinstance(variable, VariableCategorielle):
                champ = ttk.Combobox(groupe, textvariable=valeur, width=26, state="readonly",
                                     values=variable.valeurs, font=(POLICE, 12))
                aide = f"{len(variable.valeurs)} choix possibles"
            else:
                continue
            champ.grid(row=2 * ligne + 1, column=0, sticky="ew")
            tk.Label(groupe, text=aide, bg="white", fg=TEXTE_SECONDAIRE,
                     font=(POLICE, 10)).grid(row=2 * ligne + 1, column=1, sticky="w", padx=(8, 0))

        ttk.Button(cadre, text="Prédire", style="Principal.TButton",
                   command=self._sur_prediction).pack(fill="x", pady=(14, 6))
        ttk.Button(cadre, text="Voir l'historique", style="Secondaire.TButton",
                   command=self._afficher_historique).pack(fill="x")

        meta = self._application.metadonnees
        resume = (f"Modèle : {meta['modele']}\n"
                  f"Score F1 sur des courses jamais vues : {meta['scores_test']['f1']:.2f}\n"
                  f"(règle simple « départ dans le top 3 » : "
                  f"{meta['references_f1']['depart_top3']:.2f})")
        tk.Label(cadre, text=resume, bg=FOND, fg=TEXTE_SECONDAIRE, font=(POLICE, 11),
                 justify="left").pack(anchor="w", pady=(14, 0))
        tk.Label(cadre, text=AVERTISSEMENT, bg=FOND, fg=ROUGE_F1, font=(POLICE, 10),
                 wraplength=360, justify="left").pack(anchor="w", pady=(10, 0))

    def _construire_carte_resultat(self, cadre: tk.Frame) -> None:
        """Grande carte colorée qui affiche le verdict, le pourcentage et une barre."""
        self._carte = tk.Frame(cadre, bg=GRIS_CLAIR, highlightthickness=0)
        self._carte.pack(fill="x")
        interieur = tk.Frame(self._carte, bg=GRIS_CLAIR)
        interieur.pack(fill="x", padx=20, pady=16)
        self._elements_carte = [self._carte, interieur]

        self._etiquette_verdict = tk.Label(
            interieur, text="En attente d'une prédiction", bg=GRIS_CLAIR, fg=GRIS,
            font=(POLICE, 24, "bold"), anchor="w")
        self._etiquette_verdict.pack(fill="x")
        ligne = tk.Frame(interieur, bg=GRIS_CLAIR)
        ligne.pack(fill="x", pady=(4, 0))
        self._etiquette_pourcentage = tk.Label(ligne, text="– %", bg=GRIS_CLAIR, fg=GRIS,
                                               font=(POLICE, 44, "bold"))
        self._etiquette_pourcentage.pack(side="left")
        self._etiquette_detail = tk.Label(
            ligne, text="Remplissez le formulaire à gauche\npuis cliquez sur « Prédire ».",
            bg=GRIS_CLAIR, fg=TEXTE_SECONDAIRE, font=(POLICE, 13), justify="left")
        self._etiquette_detail.pack(side="left", padx=(16, 0))
        self._elements_carte += [ligne]

        self._barre = tk.Canvas(interieur, height=14, bg="white", highlightthickness=0)
        self._barre.pack(fill="x", pady=(10, 0))

    def _mettre_a_jour_carte(self, resultat: ResultatPrediction) -> None:
        """Colore la carte et remplit le verdict, le pourcentage et la barre."""
        fond, couleur = (VERT_CLAIR, VERT) if resultat.podium_predit else (GRIS_CLAIR, GRIS)
        for element in self._elements_carte:
            element.configure(bg=fond)
        icone = "🏆 " if resultat.podium_predit else ""
        self._etiquette_verdict.configure(text=icone + resultat.verdict, bg=fond, fg=couleur)
        self._etiquette_pourcentage.configure(text=f"{resultat.probabilite:.0%}", bg=fond,
                                              fg=couleur)
        comparaison = "plus" if resultat.probabilite >= resultat.taux_moyen else "moins"
        self._etiquette_detail.configure(
            bg=fond,
            text=(f"de chances de finir dans les 3 premiers\n"
                  f"C'est {comparaison} que la moyenne de tous les pilotes "
                  f"({resultat.taux_moyen:.0%})."))
        self._barre.update_idletasks()
        largeur = self._barre.winfo_width()
        self._barre.delete("all")
        self._barre.create_rectangle(0, 0, largeur * resultat.probabilite, 14,
                                     fill=couleur, width=0)
        x_seuil = largeur * resultat.seuil
        self._barre.create_line(x_seuil, 0, x_seuil, 14, fill=TEXTE, width=2)

    def _construire_graphique(self, cadre: tk.Frame) -> None:
        """Emplacement du graphique Matplotlib, sous la carte de résultat."""
        self._cadre_graphique = tk.Frame(cadre, bg="white")
        self._cadre_graphique.pack(fill="both", expand=True, pady=(16, 0))
        figure = Figure(figsize=(6.5, 4))
        axe = figure.add_subplot()
        axe.text(0.5, 0.5, "Le graphique apparaîtra après la première prédiction.",
                 ha="center", va="center", color=TEXTE_SECONDAIRE)
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
        self._mettre_a_jour_carte(resultat)
        position = int(saisies["position_depart"].strip())
        self._afficher_figure(figure_courbe_probabilite(
            positions, probabilites, position, resultat.probabilite, resultat.seuil))

    def _afficher_historique(self) -> None:
        """Ouvre une fenêtre avec le tableau des prédictions passées."""
        lignes = self._application.historique()
        fenetre = tk.Toplevel(self._racine)
        self._forcer_mode_clair(fenetre)
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
