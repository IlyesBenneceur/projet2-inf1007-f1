import json
import shutil

import pytest

from src.application import Application, ResultatPrediction
from src.exceptions import DonneeInvalideError
from src.historique import Historique

SAISIES_VALIDES = {"position_depart": "2", "annee": "2005", "ecurie": "ferrari",
                   "motoriste": "ferrari", "manufacturier_pneus": "pirelli",
                   "circuit": "monza", "type_circuit": "RACE"}


@pytest.fixture
def application(dossier_modele_entraine, tmp_path):
    return Application(dossier_modele_entraine, str(tmp_path / "historique.csv"))


def test_prediction_valide_ajoutee_a_l_historique(application):
    resultat = application.predire(SAISIES_VALIDES)
    assert 0 <= resultat.probabilite <= 1
    historique = application.historique()
    assert len(historique) == 1
    assert historique[0]["ecurie"] == "ferrari"
    assert float(historique[0]["probabilite"]) == pytest.approx(resultat.probabilite, abs=1e-4)


def test_saisies_invalides_regroupees_et_rien_n_est_enregistre(application):
    mauvaises = {**SAISIES_VALIDES, "position_depart": "abc", "ecurie": ""}
    with pytest.raises(DonneeInvalideError) as erreur:
        application.predire(mauvaises)
    message = str(erreur.value)
    assert "Position de départ" in message and "Écurie" in message
    assert application.historique() == []


def test_courbe_couvre_toutes_les_positions(application):
    positions, probabilites = application.courbe_selon_position(SAISIES_VALIDES)
    assert list(positions) == list(range(1, 11))
    assert probabilites.shape == (10,)


def test_verdict_selon_le_seuil():
    assert ResultatPrediction(0.40, 0.30, 0.14).verdict == "Podium probable"
    assert ResultatPrediction(0.20, 0.30, 0.14).verdict == "Podium peu probable"
    assert "plus que la moyenne" in str(ResultatPrediction(0.20, 0.30, 0.14))


def test_historique_vide_puis_trie_du_plus_recent(tmp_path):
    historique = Historique(str(tmp_path / "h.csv"), ["x"])
    assert historique.lire() == []
    for x in (1, 2, 3):  # trois ajouts dans la même seconde
        historique.ajouter({"x": x}, 0.1, "a")
    assert len(historique) == 3
    assert [ligne["x"] for ligne in historique.lire()] == ["3", "2", "1"]


def test_historique_illisible_leve_une_erreur_claire(tmp_path):
    chemin = tmp_path / "h.csv"
    chemin.write_bytes("horodatage,écurie\n2026-01-01T00:00:00,é\n".encode("cp1252"))
    with pytest.raises(DonneeInvalideError):
        Historique(str(chemin), ["x"]).lire()


def _copier_modele(source, tmp_path):
    """Copie le dossier du modèle entraîné pour pouvoir l'abîmer dans un test."""
    copie = tmp_path / "copie"
    shutil.copytree(source, copie)
    return copie


@pytest.mark.parametrize("cle", ["variables", "taux_podium", "scores_test"])
def test_metadonnees_incompletes_refusees(dossier_modele_entraine, tmp_path, cle):
    copie = _copier_modele(dossier_modele_entraine, tmp_path)
    chemin = copie / "metadonnees.json"
    meta = json.loads(chemin.read_text(encoding="utf-8"))
    del meta[cle]
    chemin.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(DonneeInvalideError, match=cle):
        Application(str(copie), str(tmp_path / "h.csv"))


def test_metadonnees_qui_ne_sont_pas_un_dictionnaire(dossier_modele_entraine, tmp_path):
    copie = _copier_modele(dossier_modele_entraine, tmp_path)
    (copie / "metadonnees.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DonneeInvalideError):
        Application(str(copie), str(tmp_path / "h.csv"))


def test_position_decimale_acceptee_et_convertie_en_entier(application):
    valeurs = application.valider({**SAISIES_VALIDES, "position_depart": "2.0"})
    assert valeurs["position_depart"] == 2
    assert isinstance(valeurs["position_depart"], int)