import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_random_forest_classification_accuracy():
    """
    Sprawdza, czy model RandomForestClassifier potrafi się przetrenować
    na zbiorze Breast Cancer i osiągnąć dokładność powyżej 85%.
    """
    # 1. PRZYGOTOWANIE DANYCH (Arrange)

    data = load_breast_cancer()
    X = data.data  # Cechy (np. wielkość guza)
    y = data.target  # Etykiety (0 - złośliwy, 1 - łagodny)

    # Dzielimy dane na zbiór treningowy (80%) i testowy (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)

    # 2. WYKONANIE AKCJI (Act)
    model.fit(X_train, y_train)

    # Generujemy przewidywania dla danych testowych, których model nie widział
    predictions = model.predict(X_test)

    # Obliczamy dokładność (ile przewidywań było trafnych)
    accuracy = accuracy_score(y_test, predictions)

    # 3. WERYFIKACJA (Assert)
    # Sprawdzamy czy model w ogóle coś zwrócił
    assert len(predictions) == len(y_test), "Model zwrócił złą liczbę predykcji!"

    # Sprawdzamy, czy dokładność wynosi minimum 85%
    assert accuracy >= 0.85, f"Dokładność modelu jest zbyt niska: {accuracy * 100}%"