import pytest
from sklearn.ensemble import RandomForestClassifier

def test_fit_on_empty_dataset():
    """
    Sprawdza, czy próba trenowania modelu na całkowicie pustych listach
    (brak danych) zostaje bezpiecznie przechwycona i rzuca wyjątek ValueError.
    """
    # 1. ARRANGE (Przygotowanie pustych danych i modelu)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_empty = []
    y_empty = []
    
    # 2 & 3. ACT & ASSERT (Wykonanie i weryfikacja błędu)
    with pytest.raises(ValueError):
        model.fit(X_empty, y_empty)

def test_fit_shape_mismatch():
    """
    Sprawdza odporność algorytmu na błędne wymiary danych wejściowych.
    Upewnia się, że podanie różnej liczby próbek (X) i etykiet (y)
    skutkuje wyrzuceniem błędu ValueError.
    """
    # 1. ARRANGE (Przygotowanie zepsutych danych)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # 3 wiersze danych (cech), ale tylko 2 odpowiedzi (etykiety)
    X_bad_shape = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y_bad_shape = [0, 1]
    
    # 2 & 3. ACT & ASSERT (Sprawdzenie, czy model to zablokuje)
    with pytest.raises(ValueError):
        model.fit(X_bad_shape, y_bad_shape)