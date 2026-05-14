import time
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

def test_random_forest_parallelization():
    # Duże zbiory danych
    print("\nGenerowanie danych...")
    X, y = make_classification(n_samples=100000, n_features=20, random_state=42)

    # Pojedynczy rdzeń (pomiar czasu)
    clf_single = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    start_time = time.time()
    clf_single.fit(X, y)
    duration_single = time.time() - start_time
    print(f"Czas treningu (n_jobs=1): {duration_single:.2f}s")

    # Wszystkie rdzenie (pomiar czasu)
    clf_parallel = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    start_time = time.time()
    clf_parallel.fit(X, y)
    duration_parallel = time.time() - start_time
    print(f"Czas treningu (n_jobs=-1): {duration_parallel:.2f}s")

    # Logowanie wyników, asercja
    print(f"Przyspieszenie: {duration_single / duration_parallel:.2f}x")
    
    # Czy wersja równoległa jest szybsza??
    assert duration_parallel < duration_single, "Trening równoległy nie jest szybszy od jednowątkowego!"
if __name__ == "__main__":
    test_random_forest_parallelization()
