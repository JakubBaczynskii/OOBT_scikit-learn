# Pamiętaj, żeby dodać te dwa importy na górze pliku, jeśli ich tam nie ma:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ... (tutaj jest Twój poprzedni test test_random_forest_regression_performance) ...

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