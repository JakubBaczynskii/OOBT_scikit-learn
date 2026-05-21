import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def test_pipeline_integration():
    """
    Testuje pełny przepływ danych (Pipeline) w scikit-learn,
    łącząc normalizację (StandardScaler) z klasyfikacją.
    """
    # 1. ARRANGE: Przygotowanie danych
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # 2. Tworzenie Pipeline'u
    # Pierwszy krok: standaryzacja danych (skalowanie do średniej 0 i wariancji 1)
    # Drugi krok: właściwy model klasyfikacyjny
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])