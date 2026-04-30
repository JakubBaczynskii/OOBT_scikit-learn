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