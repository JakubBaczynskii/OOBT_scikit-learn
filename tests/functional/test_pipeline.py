import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import make_classification


def test_pipeline_scaling_and_classification():
    """
    Sprawdza, czy obiekt Pipeline poprawnie łączy skalowanie danych
    z modelem klasyfikacyjnym i czy generuje predykcje bez błędów.
    """
    # 1. ARRANGE (Przygotowanie)
    # Generujemy malutki, syntetyczny zbiór danych (100 wierszy, 5 kolumn)
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Tworzymy potok: najpierw skaler, potem model (Support Vector Classifier)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', random_state=42))
    ])

    # 2. ACT (Wykonanie)
    # Uczymy cały potok naraz (dane są najpierw skalowane, a potem uczony jest model)
    pipeline.fit(X, y)

    # Przewidujemy wyniki dla tych samych danych
    predictions = pipeline.predict(X)

    # 3. ASSERT (Weryfikacja)
    # Sprawdzamy, czy pipeline wyrzucił tyle predykcji, ile podaliśmy wierszy
    assert len(predictions) == 100, "Pipeline zwrócił złą liczbę przewidywań!"

    # Upewniamy się, że predykcje to tylko klasy 0 i 1 (zgodnie z make_classification)
    assert set(predictions).issubset({0, 1}), "Predykcje zawierają nieznane klasy!"