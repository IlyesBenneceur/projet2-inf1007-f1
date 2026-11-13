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
    historique.ajouter({"x": 1}, 0.1, "a")
    historique.ajouter({"x": 2}, 0.2, "b")
    lignes = historique.lire()
    assert len(historique) == 2
    assert lignes[0]["horodatage"] >= lignes[1]["horodatage"]
