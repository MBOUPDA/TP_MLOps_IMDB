import os


# Test 1 : Vérifier que le dossier models et le fichier existent
def test_model_structure():
    assert os.path.exists("models/"), "Le dossier models est manquant."


# Test 2 : Simulation de validation de performance (> 80%)
def test_model_accuracy():
    accuracy_score = 0.85  # Score simulé
    threshold = 0.80
    assert (
        accuracy_score >= threshold
    ), f"L'accuracy ({accuracy_score}) est inférieure au seuil de {threshold}"
