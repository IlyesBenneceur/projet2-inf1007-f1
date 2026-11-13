import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.donnees import COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES
from src.exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
from src.modele import (ModelePodium, chronometrer, choisir_seuil, f1_numpy, lire_metadonnees,
                        predire_reference_grille, predire_reference_majoritaire)


def test_f1_numpy_identique_a_scikit_learn():
    generateur = np.random.default_rng(1)
    vrai = generateur.integers(0, 2, 1000)
    predit = generateur.integers(0, 2, 1000)
    assert f1_numpy(vrai, predit) == pytest.approx(f1_score(vrai, predit))


def test_f1_numpy_cas_simple_calcule_a_la_main():
    # VP = 1, FP = 1, FN = 1  ->  F1 = 2 / (2 + 1 + 1) = 0.5
    assert f1_numpy(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == pytest.approx(0.5)


def test_f1_numpy_sans_aucun_positif_vaut_zero():
    zeros = np.zeros(5, dtype=int)
    assert f1_numpy(zeros, zeros) == 0.0


def test_f1_numpy_tailles_differentes():
    with pytest.raises(DonneeInvalideError):
        f1_numpy(np.array([1, 0]), np.array([1]))


def test_references():
    assert list(predire_reference_majoritaire(pd.Series([0, 0, 1]), 4)) == [0, 0, 0, 0]
    assert list(predire_reference_grille(pd.Series([1, 3, 4, 20]))) == [1, 1, 0, 0]


def test_choisir_seuil_maximise_le_f1():
    vrai = np.array([1, 1, 0, 0])
    probabilites = np.array([0.35, 0.40, 0.20, 0.10])
    # Avec le seuil 0.5 on ne prédit aucun podium ; le meilleur seuil est entre 0.2 et 0.35.
    assert 0.2 < choisir_seuil(vrai, probabilites) <= 0.35


def test_decorateur_chronometrer_conserve_le_resultat():
    @chronometrer
    def addition(a, b):
        return a + b

    assert addition(2, 3) == 5
    assert addition.derniere_duree >= 0


def test_modele_non_entraine_refuse_de_predire():
    modele = ModelePodium(COLONNES_NUMERIQUES, COLONNES_CATEGORIELLES)
    with pytest.raises(ModeleNonEntraineError):
        modele.predire(pd.DataFrame())


def test_sauvegarde_puis_rechargement(dossier_modele_entraine, tmp_path):
    modele = ModelePodium.charger(f"{dossier_modele_entraine}/modele_podium.joblib")
    assert modele.est_entraine
    exemple = pd.DataFrame([{"position_depart": 1, "annee": 2005, "ecurie": "ferrari",
                             "motoriste": "ferrari", "manufacturier_pneus": "pirelli",
                             "circuit": "monza", "type_circuit": "RACE"}])
    probabilite = modele.probabilites(exemple)[0]
    assert 0 <= probabilite <= 1
    copie = tmp_path / "copie.joblib"
    modele.sauvegarder(str(copie))
    assert ModelePodium.charger(str(copie)).probabilites(exemple)[0] == pytest.approx(probabilite)


def test_partir_devant_augmente_la_probabilite(dossier_modele_entraine):
    modele = ModelePodium.charger(f"{dossier_modele_entraine}/modele_podium.joblib")
    base = {"annee": 2005, "ecurie": "ferrari", "motoriste": "ferrari",
            "manufacturier_pneus": "pirelli", "circuit": "monza", "type_circuit": "RACE"}
    devant, derriere = modele.probabilites(pd.DataFrame([{**base, "position_depart": 1},
                                                          {**base, "position_depart": 10}]))
    assert devant > derriere


def test_metadonnees_contiennent_scores_et_variables(dossier_modele_entraine):
    meta = lire_metadonnees(f"{dossier_modele_entraine}/metadonnees.json")
    assert {v["nom"] for v in meta["variables"]} == set(COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES)
    assert 0 <= meta["scores_test"]["f1"] <= 1
    assert meta["references_f1"]["majoritaire"] == 0.0


def test_fichiers_absents_ou_invalides(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        ModelePodium.charger(str(tmp_path / "absent.joblib"))
    with pytest.raises(FichierIntrouvableError):
        lire_metadonnees(str(tmp_path / "absent.json"))
    mauvais = tmp_path / "mauvais.json"
    mauvais.write_text("{pas du json", encoding="utf-8")
    with pytest.raises(DonneeInvalideError):
        lire_metadonnees(str(mauvais))
