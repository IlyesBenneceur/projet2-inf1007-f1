# Prédicteur de podium F1 — Projet 2 INF1007

Application qui estime la probabilité qu'un pilote de Formule 1 termine **sur le podium**
(dans les 3 premiers) à partir d'informations connues **avant le départ** : sa position
sur la grille, l'année, l'écurie, le motoriste, le manufacturier de pneus et le circuit.
Un modèle de régression logistique est entraîné une fois, puis une interface Tkinter
permet de l'interroger.

> Exercice pédagogique : l'outil n'est pas une aide à la décision et ne doit pas servir à parier.

## Jeu de données

- **Source :** [F1DB](https://github.com/f1db/f1db), fichiers `f1db-races-race-results.csv`
  et `f1db-races.csv` (version v2026.20.0).
- **Licence :** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). La redistribution
  est permise avec attribution.
- **Copie publiée :** `donnees/f1_resultats.csv` (27 731 lignes, 1,9 Mo), produite par
  `outils/extraire_f1db.py`. Les colonnes qui identifient les pilotes (`driverId`,
  `driverNumber`) ont été retirées. Un pilote reste retrouvable en croisant la course, l'écurie
  et la position avec les résultats officiels : ce sont des personnalités publiques et des
  résultats déjà publiés, et l'application ne manipule aucun nom.
- **Caractéristiques :**
  - numériques : `position_depart`, `annee` ;
  - catégorielles : `ecurie`, `motoriste`, `manufacturier_pneus`, `circuit`, `type_circuit`.
- **Cible :** `podium`, qui vaut 1 si le pilote termine 1er, 2e ou 3e (classification binaire).

**Exploration** (types, valeurs manquantes, valeurs aberrantes) :

- **Types :** l'année et les identifiants de course sont des entiers ; les positions sont des
  nombres à virgule, parce qu'elles peuvent être vides ; l'écurie, le motoriste, les pneus et
  le circuit sont du texte ; `sharedCar` est un booléen.
- **Valeurs manquantes :** `positionNumber` est vide pour 10 938 lignes : ce sont les pilotes
  non classés (abandons, disqualifications, non-partants). `gridPositionNumber` est vide pour
  2 017 lignes, dont 244 départs depuis la voie des stands (`PL`).
- **Valeurs aberrantes :** 150 lignes ont une position de départ au-delà de 26 (de 1950 à
  1986, dont 77 à l'Indianapolis 500). Elles sont réelles, car les grilles étaient plus
  grandes à l'époque : nous les gardons.

**Nettoyage** (voir `src/donnees.py`) :

- On retire les pilotes qui n'ont pas pris le départ (DNQ, DNPQ, DNS, DNP, EX).
- On retire les 127 lignes de voitures partagées par deux pilotes (années 1950), qui créent
  des doublons.
- Un pilote parti de la voie des stands reçoit la dernière place de la grille + 1.
- On retire les 29 lignes sans position de départ connue : 26 lignes Tyrrell de 1984
  (disqualifiée pour toute la saison), 2 lignes en 1990 et 1 en 2021.

Il reste **25 794 lignes**, dont 13,6 % de podiums.

## Installation et lancement

```bash
git clone https://github.com/IlyesBenneceur/projet2-inf1007-f1.git
cd projet2-inf1007-f1
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt

python -m src.main                 # lance l'interface (le modèle est déjà dans modeles/)
python -m src.main --entrainer     # (optionnel) ré-entraîne le modèle et régénère les graphiques
python -m pytest                   # lance les tests (ou simplement : pytest)
```

Les commandes se lancent depuis la racine du dépôt, avec `python -m`. `python src/main.py`
ne fonctionne pas, car le code importe le paquet `src`.

Le projet a été testé avec Python 3.14. Tkinter est inclus avec Python. Sous Linux, il peut
falloir installer `python3-tk`.

## Structure du code

```
src/
├── exceptions.py    # ProjetError → DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
├── donnees.py       # lecture du CSV (générateur par blocs), nettoyage, classe JeuDeDonnees
├── variables.py     # Variable (abstraite) → VariableNumerique, VariableCategorielle
├── modele.py        # ModelePodium, F1 en NumPy, références, décorateur chronometrer, JSON
├── graphiques.py    # figures Matplotlib (exploration, matrice de confusion, courbe de l'interface)
├── historique.py    # Historique : ajout et lecture des prédictions (CSV)
├── application.py   # Application : contient le modèle, les variables et l'historique
├── entrainement.py  # chaîne complète : données → modèle → fichiers sauvegardés
├── interface.py     # fenêtre Tkinter (affichage seulement, aucune logique)
└── main.py          # point d'entrée
modeles/             # modele_podium.joblib + metadonnees.json
graphiques/          # graphiques enregistrés avec savefig
historique/          # predictions.csv (créé à la première prédiction, non versionné)
outils/              # script d'extraction de F1DB
tests/               # tests pytest
```

**Conception orientée objet**

- **Héritage et polymorphisme :** `VariableNumerique` et `VariableCategorielle` héritent de
  la classe abstraite `Variable` et redéfinissent `valider()`. `Application.valider()`
  appelle la même méthode sur chaque variable, sans connaître son type.
- **Composition :** `Application` contient un `ModelePodium`, une liste de `Variable` et un
  `Historique`.
- **Encapsulation :** les attributs de `JeuDeDonnees`, `ModelePodium`, `Variable`,
  `Historique` et `Application` sont protégés (`_seuil`, `_valeurs`, `_pipeline`, etc.) et
  exposés en lecture seulement par des propriétés. `ResultatPrediction`, un simple
  conteneur de résultat, garde des attributs publics.
- **Méthodes spéciales :** `__eq__` et `__str__` dans `Variable`, `__repr__` dans ses deux
  classes enfants ; `__len__` et `__repr__` dans `JeuDeDonnees` ; `__len__` dans `Historique` ;
  `__str__` dans `ResultatPrediction`.
- **Séparation calcul / affichage :** `interface.py` ne fait qu'appeler `Application`. Toute
  la logique est testable sans fenêtre.
- **Gestion des erreurs :** chacun de ces cas produit un message clair, jamais un traceback :
  - une saisie vide, non numérique ou hors intervalle ;
  - un historique impossible à écrire (par exemple `predictions.csv` ouvert dans Excel) ou
    illisible ;
  - un modèle `.joblib` absent ou corrompu ;
  - des métadonnées JSON absentes, mal formées ou incomplètes ;
  - un CSV de données absent, vide ou illisible.

**Outils du chapitre 7**

- **Générateur :** `lire_par_blocs()` lit le CSV par blocs de 5 000 lignes avec `yield`.
  `charger_csv()` assemble ensuite les blocs : avec 27 731 lignes, tout le fichier tient en
  mémoire. Le même générateur permettrait de traiter bloc par bloc un fichier trop gros
  pour la mémoire.
- **Décorateur :** `@chronometrer` mesure la durée de l'entraînement, enregistrée dans les
  métadonnées.
- **Lambda :** clé de tri des valeurs catégorielles, sans tenir compte des majuscules, et
  choix du meilleur seuil (`max` avec `key=lambda`).

**NumPy :** `f1_numpy()` calcule le F1 de façon vectorisée. Un test vérifie qu'il donne le
même résultat que `sklearn.metrics.f1_score`.

## Modèle et résultats

- **Modèle :** régression logistique. Les variables numériques sont standardisées et les
  variables catégorielles encodées en one-hot.
- **Séparation :** 80 % entraînement et 20 % test (`train_test_split`, graine 42, stratifié).
- **Métrique :** F1 de la classe « podium ». Les classes sont déséquilibrées : seulement
  13,6 % de podiums. Un modèle qui répond toujours « pas de podium » aurait 86 %
  d'exactitude sans rien prédire d'utile. L'exactitude n'est donc pas une bonne métrique ici.
- **Seuil de décision :** il est choisi par validation croisée à 5 plis, sur l'entraînement
  seulement. Le seuil retenu est 0,30 au lieu de 0,5.

| Modèle (ensemble de test, 5 159 lignes) | F1 |
|---|---|
| Référence : toujours « pas de podium » | 0,000 |
| Référence : « podium si départ dans le top 3 » | 0,549 |
| **Régression logistique (seuil 0,30)** | **0,579** |

Précision 0,49, rappel 0,72, exactitude 0,86. La matrice de confusion est dans
`graphiques/matrice_confusion.png`.

**Analyse honnête**

- Le modèle bat la référence triviale, mais ne fait qu'environ 3 points de mieux que la règle
  de bon sens « départ dans le top 3 ». La position de départ porte presque toute
  l'information (voir `graphiques/exploration_taux_podium.png`).
- Avec le seuil de 0,5, le modèle faisait *moins bien* que cette règle (F1 = 0,46). C'est le
  choix du seuil qui le fait passer devant.
- Ce qui reste imprévisible avant la course (abandons, accidents, météo, stratégie) limite
  fortement le score possible.
- **Limites :**
  - la séparation aléatoire mélange les époques, de 1950 à 2026 ;
  - une écurie rare, ou une nouvelle écurie, est mal estimée ;
  - le modèle ne connaît ni le pilote, ni la forme du moment ;
  - la régression logistique additionne l'effet de l'écurie et celui de l'année : une écurie
    a donc la même force à toutes les époques. À Monza, départ en 5e position, Mercedes 2014
    et Mercedes 2024 obtiennent toutes deux 41 % ;
  - chaque saisie est validée séparément, pas leur combinaison : l'interface accepte une
    voiture qui n'a jamais existé (Ferrari à moteur Honda en 1950, en pole position : 82 %).

## Tests et couverture

62 tests pytest, répartis en 5 fichiers :

- `test_donnees.py` ;
- `test_variables.py` ;
- `test_modele.py` ;
- `test_application.py` ;
- `test_exceptions.py`.

Ils couvrent des cas usuels, des cas limites (bornes des intervalles, fichier vide, aucun
positif, deux départs depuis la voie des stands dans la même course) et des cas d'erreur
(`pytest.raises`), dont des fichiers du modèle et un historique corrompus. Les tests
utilisent un petit jeu synthétique (`tests/conftest.py`) et non le vrai CSV. La CI GitHub
Actions les lance à chaque `push`.

```bash
python -m coverage run --source=src -m pytest
python -m coverage report --omit=src/interface.py,src/main.py
```

**Couverture de la logique (hors interface et main) : 94 %.** En incluant `interface.py` et
`main.py`, qui ne sont pas testés automatiquement, la couverture est de 66 %.

## Documentation officielle consultée

- pandas : [`read_csv` (paramètre `chunksize`)](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
  et [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- scikit-learn :
  - [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) ;
  - [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) ;
  - [`cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) ;
  - [Model persistence (joblib)](https://scikit-learn.org/stable/model_persistence.html).
- Tkinter : [tkinter — Python interface to Tcl/Tk](https://docs.python.org/3/library/tkinter.html)
  et [tkinter.ttk](https://docs.python.org/3/library/tkinter.ttk.html)
- Matplotlib : [Embedding in Tk](https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html)

## Documents du projet

- [DECISIONS.md](DECISIONS.md) : fiche de décisions.
- [JOURNAL_IA.md](JOURNAL_IA.md) : journal d'utilisation de l'IA.