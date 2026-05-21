import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier


def test_sparse_performance():
    #Tworzenie datasetu
    X, y = make_classification(
        n_samples=5000,
        n_features=2000,
        random_state=42
    )

    #Ustawienie większości wartości na 0
    mask = np.random.rand(*X.shape) < 0.95
    X[mask] = 0

    #Dense i sparse
    X_dense = X
    X_sparse = csr_matrix(X)

    #Model dla dense
    model_dense = SGDClassifier(random_state=42)

    start_dense = time.time()
    model_dense.fit(X_dense, y)
    dense_time = time.time() - start_dense

    #Model dla sparse
    model_sparse = SGDClassifier(random_state=42)

    start_sparse = time.time()
    model_sparse.fit(X_sparse, y)
    sparse_time = time.time() - start_sparse

    #Wyniki
    print("\nDense time:", dense_time)
    print("Sparse time:", sparse_time)

    #Sprawdzenie czy model działa
    assert model_sparse.coef_ is not None
