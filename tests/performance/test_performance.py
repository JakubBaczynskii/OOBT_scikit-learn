import time
import pytest
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fit_performance_small_dataset():
    # Zadanie: 1000 wierszy
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    start_time = time.perf_counter()
    model.fit(X, y)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    logger.info(f"\n[SMALL DATASET] Training time for 1000 rows: {duration:.4f} seconds")
    
    assert duration > 0
def test_fit_performance_large_dataset():
    # Zadanie: 50000 wierszy
    X, y = make_classification(n_samples=50000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    start_time = time.perf_counter()
    model.fit(X, y)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    logger.info(f"\n[LARGE DATASET] Training time for 50000 rows: {duration:.4f} seconds")
    
    assert duration > 0

def test_performance_scaling():
    # Przygotowanie danych
    X_small, y_small = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_large, y_large = make_classification(n_samples=50000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    # Pomiar mały
    t1 = time.perf_counter()
    model.fit(X_small, y_small)
    duration_small = time.perf_counter() - t1

    # Pomiar duży
    t2 = time.perf_counter()
    model.fit(X_large, y_large)
    duration_large = time.perf_counter() - t2

    logger.info(f"Scaling factor: {duration_large / duration_small:.2f}x")
    
    # Duży zbiór (50x więcej danych) powinien zająć wyraźnie więcej czasu niż mały, ale nie mniej niż on sam.
    assert duration_large > duration_small
