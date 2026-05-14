import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

def test_linear_regression_performance():
    """
    Sprawdza działanie klasycznego modelu LinearRegression na zbiorze diabetes.
    Tym razem weryfikujemy błąd średniokwadratowy (MSE) zamiast R2.
    """
    # 1. ARRANGE
    data = load_diabetes()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    
    # 2. ACT
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Obliczamy MSE (Mean Squared Error). W przypadku MSE - im mniej, tym lepiej!
    mse = mean_squared_error(y_test, predictions)
    
    # 3. ASSERT
    # Dla zbioru diabetes klasyczna regresja liniowa osiąga MSE na poziomie ok. 2900.
    # Zabezpieczamy się testem, że błąd nie powinien przekroczyć 3500.
    assert mse < 3500, f"Błąd MSE jest zbyt duży: {mse}. Model słabo się uczy."
    assert len(predictions) == len(y_test)