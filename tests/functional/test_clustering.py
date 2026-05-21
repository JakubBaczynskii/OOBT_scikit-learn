import pytest
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def test_kmeans_clustering_performance():
    """
    Testuje algorytm KMeans (Unsupervised Learning) na syntetycznych danych.
    Sprawdza, czy model poprawnie znajduje skupiska i sensownie je grupuje.
    """
    # 1. ARRANGE: Generowanie syntetycznego zbioru danych (Twój etap z Issue!)
    # Używamy make_blobs do stworzenia 300 punktów, ułożonych w 3 wyraźne grupy (centers=3)
    # w dwuwymiarowej przestrzeni.
    X, _ = make_blobs(
        n_samples=300, 
        centers=3, 
        cluster_std=0.60, 
        random_state=42
    )
    
    # Inicjalizacja modelu KMeans - mówimy mu, żeby poszukał 3 klastrów
    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    
    # 2. ACT: Trenowanie modelu (bez podawania odpowiedzi, bo to uczenie bez nadzoru)
    model.fit(X)
    
    # Pobieramy przypisane przez model etykiety dla każdego punktu
    predicted_labels = model.labels_
    
    # 3. ASSERT: Weryfikacja działania modelu
    # Sprawdzamy, czy model wygenerował dokładnie 3 centra klastrów, a każdy ma 2 współrzędne (X, Y)
    assert model.cluster_centers_.shape == (3, 2), "Model nie utworzył poprawnych centrów klastrów."
    
    # Obliczamy jakość klastryzacji za pomocą silhouette_score
    # Wynik z przedziału od -1 do 1. Im bliżej 1, tym klastry są lepiej odseparowane od siebie.
    score = silhouette_score(X, predicted_labels)
    
    # Oczekujemy, że przy tak prostych danych (make_blobs), wynik będzie wysoki (znacznie powyżej 0)
    assert score > 0.5, f"Silhouette score jest zbyt niski: {score}. Klastry są słabo rozdzielone."