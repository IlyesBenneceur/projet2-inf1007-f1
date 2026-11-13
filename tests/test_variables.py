import pytest

from src.exceptions import DonneeInvalideError
from src.variables import VariableCategorielle, VariableNumerique, variable_depuis_metadonnees


@pytest.fixture
def depart():
    return VariableNumerique("position_depart", "Position de départ", 1, 34, entier=True)


@pytest.fixture
def ecurie():
    return VariableCategorielle("ecurie", "Écurie", ["williams", "Ferrari", "mclaren"])


def test_nombre_valide_converti(depart):
    assert depart.valider(" 3 ") == 3


def test_bornes_de_l_intervalle_acceptees(depart):
    assert depart.valider("1") == 1
    assert depart.valider("34") == 34


@pytest.mark.parametrize("saisie", ["", "   ", "trois", "0", "35", "2.5", "-1"])
def test_saisies_numeriques_invalides(depart, saisie):
    with pytest.raises(DonneeInvalideError):
        depart.valider(saisie)


def test_message_clair_hors_intervalle(depart):
    with pytest.raises(DonneeInvalideError, match="entre 1 et 34"):
        depart.valider("50")


def test_virgule_decimale_acceptee():
    note = VariableNumerique("note", "Note", 0.0, 10.0)
    assert note.valider("7,5") == pytest.approx(7.5)


def test_intervalle_inverse_refuse():
    with pytest.raises(DonneeInvalideError):
        VariableNumerique("x", "X", 10, 1)


def test_categorielle_valide_et_tri_sans_casse(ecurie):
    assert ecurie.valider("mclaren") == "mclaren"
    assert ecurie.valeurs == ["Ferrari", "mclaren", "williams"]


@pytest.mark.parametrize("saisie", ["", "red-bull"])
def test_categorielle_invalide(ecurie, saisie):
    with pytest.raises(DonneeInvalideError):
        ecurie.valider(saisie)


def test_polymorphisme_meme_methode_comportements_differents(depart, ecurie):
    variables = [depart, ecurie]
    assert [v.valider(s) for v, s in zip(variables, ["5", "williams"])] == [5, "williams"]


def test_aller_retour_metadonnees_et_egalite(depart, ecurie):
    for variable in (depart, ecurie):
        assert variable_depuis_metadonnees(variable.vers_metadonnees()) == variable
    assert depart != ecurie
    assert str(ecurie) == "Écurie"


def test_metadonnees_invalides():
    with pytest.raises(DonneeInvalideError, match="inconnu"):
        variable_depuis_metadonnees({"type": "date", "nom": "x", "libelle": "X"})
    with pytest.raises(DonneeInvalideError, match="manquante"):
        variable_depuis_metadonnees({"type": "numerique", "nom": "x"})
