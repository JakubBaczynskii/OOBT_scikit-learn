import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def test_random_forest_regression_performance():
    """
    Testuje, czy model RandomForestRegressor potrafi nauczyć się zależności
    w zbiorze danych 'diabetes' i osiągnąć wynik R2 powyżej 0.4.
    """
    # 1. PRZYGOTOWANIE DANYCH (Arrange)
    # Wczytujemy zbiór danych o cukrzycy (zadanie: przewidzieć postęp choroby)
    data = load_diabetes()
    X = data.data
    y = data.target
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Inicjalizacja modelu regresji
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 2. WYKONANIE AKCJI (Act)
    # Trenujemy model
    model.fit(X_train, y_train)
    
    # Przewidujemy wartości dla zbioru testowego
    predictions = model.predict(X_test)
    
    # Obliczamy współczynnik determinacji R2 (im bliżej 1.0, tym lepiej)
    r2 = r2_score(y_test, predictions)
    
    # 3. WERYFIKACJA (Assert)
    # Sprawdzamy, czy model faktycznie się czegoś nauczył (R2 > 0.4 to rozsądny próg dla tego zbioru)
    assert r2 > 0.4, f"Wynik R2 jest zbyt niski: {r2}. Model regresji nie działa poprawnie."
    
    # Sprawdzamy, czy liczba przewidywań zgadza się z liczbą próbek testowych
    assert len(predictions) == len(y_test)