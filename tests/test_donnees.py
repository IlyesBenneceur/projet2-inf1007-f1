import pandas as pd
import pytest

from src.donnees import (COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES, JeuDeDonnees, charger_csv,
                         imputer_depart_voie_des_stands, lire_par_blocs, nettoyer,
                         retirer_non_partants)
from src.exceptions import DonneeInvalideError, FichierIntrouvableError


def test_fichier_absent_leve_une_erreur_claire(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        charger_csv(str(tmp_path / "absent.csv"))


def test_le_generateur_lit_le_csv_par_blocs(chemin_csv):
    blocs = list(lire_par_blocs(chemin_csv, taille_bloc=120))
    # 300 lignes en blocs de 120 : 120 + 120 + 60
    assert [len(bloc) for bloc in blocs] == [120, 120, 60]


def test_taille_de_bloc_nulle_refusee(chemin_csv):
    with pytest.raises(DonneeInvalideError):
        next(lire_par_blocs(chemin_csv, taille_bloc=0))


def test_colonne_manquante_detectee(tmp_path, brut):
    chemin = tmp_path / "incomplet.csv"
    brut.drop(columns=["constructorId"]).to_csv(chemin, index=False)
    with pytest.raises(DonneeInvalideError, match="constructorId"):
        charger_csv(str(chemin))


def test_les_non_partants_et_voitures_partagees_sont_retires(brut):
    brut.loc[0, "positionText"] = "DNQ"
    brut.loc[1, "positionText"] = "DNS"
    brut.loc[2, "sharedCar"] = True
    brut.loc[3, "positionText"] = "DNF"  # a pris le départ : on le garde
    assert len(retirer_non_partants(brut)) == len(brut) - 3


def test_depart_voie_des_stands_place_derriere_la_grille(brut):
    # Course 1 : grille de 1 à 10 ; le pilote parti des stands doit avoir 11.
    brut.loc[0, ["gridPositionText", "gridPositionNumber"]] = ["PL", None]
    resultat = imputer_depart_voie_des_stands(brut)
    assert resultat.loc[0, "gridPositionNumber"] == 11


def test_nettoyer_cree_la_cible_podium(brut):
    brut.loc[0, ["positionText", "positionNumber"]] = ["DNF", None]
    propre = nettoyer(brut)
    assert list(propre.columns) == COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES + ["podium"]
    assert propre.loc[0, "podium"] == 0  # un abandon n'est pas un podium
    assert propre["podium"].sum() == (brut["positionNumber"] <= 3).sum()


def test_nettoyer_refuse_un_resultat_vide(brut):
    brut["positionText"] = "DNQ"
    with pytest.raises(DonneeInvalideError):
        nettoyer(brut)


def test_separation_reproductible_avec_la_graine(chemin_csv):
    jeu = JeuDeDonnees(chemin_csv)
    jeu.charger()
    x1, x2, _, _ = jeu.separer(0.2, graine=42)
    x1_bis, _, _, _ = jeu.separer(0.2, graine=42)
    assert x1.index.equals(x1_bis.index)
    assert len(x2) == pytest.approx(0.2 * len(jeu), abs=1)


def test_jeu_non_charge_leve_une_erreur():
    with pytest.raises(DonneeInvalideError):
        _ = JeuDeDonnees("peu_importe.csv").donnees


def test_valeurs_categorielles_triees_sans_doublon(chemin_csv):
    jeu = JeuDeDonnees(chemin_csv)
    jeu.charger()
    assert jeu.valeurs_categorielles()["ecurie"] == ["ferrari", "mclaren", "williams"]
    assert jeu.intervalles_numeriques()["position_depart"] == (1.0, 10.0)
    assert isinstance(jeu.donnees, pd.DataFrame)
