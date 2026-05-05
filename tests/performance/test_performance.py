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
