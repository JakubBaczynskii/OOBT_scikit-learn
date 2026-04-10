import os
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def test_model_persistence_saving_and_loading():
    """
    Weryfikuje, czy wyuczony model można zapisać na dysku,
    wczytać ponownie i czy wyniki predykcji pozostają w 100% identyczne.
    """
    # 1. ARRANGE (Przygotowanie)
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)

    # Generujemy bazowe predykcje przed zapisem (żeby mieć punkt odniesienia)
    original_predictions = model.predict(X)
    file_path = "test_model_temp.joblib"  # Nazwa pliku tymczasowego

    # 2. ACT (Wykonanie)
    # Zapisujemy model do pliku
    joblib.dump(model, file_path)

    # Udajemy, że wczytujemy go w zupełnie nowym procesie
    loaded_model = joblib.load(file_path)

    # Generujemy predykcje z odzyskanego modelu
    new_predictions = loaded_model.predict(X)

    # 3. ASSERT (Weryfikacja)
    # Sprawdzamy czy plik faktycznie powstał na dysku
    assert os.path.exists(file_path), "Plik z modelem nie został utworzony!"

    # Najważniejszy test: czy wyniki są identyczne?
    # W numpy/scikit-learn tablice porównujemy używając .all()
    assert (original_predictions == new_predictions).all(), "Wczytany model daje inne wyniki!"

    # 4. TEARDOWN (Sprzątanie)
    # Usuwamy plik, żeby nie śmiecić w repozytorium po wykonanym teście
    if os.path.exists(file_path):
        os.remove(file_path)