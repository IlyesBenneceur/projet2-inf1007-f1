import pytest

from src.exceptions import DonneeInvalideError, FichierIntrouvableError, ProjetError


def test_donnee_invalide_est_attrapee_comme_erreur_du_projet():
    with pytest.raises(ProjetError):
        raise DonneeInvalideError("saisie vide")


def test_le_message_de_l_exception_est_conserve():
    erreur = DonneeInvalideError("année hors intervalle")
    assert str(erreur) == "année hors intervalle"


def test_fichier_introuvable_n_est_pas_une_donnee_invalide():
    with pytest.raises(FichierIntrouvableError):
        raise FichierIntrouvableError("modele.joblib")
    assert not issubclass(FichierIntrouvableError, DonneeInvalideError)