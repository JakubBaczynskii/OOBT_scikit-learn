# Scenariusze Testów Akceptacyjnych (UAT) – moduł `scikit-learn`

Poniższy dokument definiuje zbiór testów akceptacyjnych oraz szczegółowych przypadków testowych (Test Cases) dla biblioteki `scikit-learn`. Testy te weryfikują stabilność i poprawność działania modułu z perspektywy użytkownika końcowego (Out Of The Box) dla najbardziej kluczowych przypadków użycia.

---

## Scenariusz 1: Rozpoznawanie nowotworów (Zdolność do klasyfikacji)

**Cel testu:** Weryfikacja zdolności modelu klasyfikacyjnego (np. `RandomForestClassifier`) do poprawnego uczenia się na rzeczywistym zbiorze danych medycznych oraz generowania trafnych predykcji.
* **Oczekiwany rezultat:** System bezbłędnie wczyta wbudowany zbiór danych `breast_cancer`, dokona podziału na zbiory treningowy i testowy, a następnie pomyślnie ukończy proces trenowania bez zgłaszania wyjątków. Następnie wygeneruje wektor predykcji.
* **Kryterium zaliczenia:** Ostateczny wynik dokładności modelu (`accuracy_score`), obliczony na zbiorze testowym, wynosi co najmniej 85%.

### Przypadki testowe (Test Cases)

**TC 1.1: Weryfikacja minimalnej skuteczności modelu (Accuracy Threshold)**
* **Cel:** Potwierdzenie, że wytrenowany model przekracza próg 85% skuteczności.
* **Warunki wstępne:** Środowisko Python z zainstalowaną biblioteką `scikit-learn`.
* **Kroki testowe:**
  1. Zaimportuj zbiór `breast_cancer` z `sklearn.datasets`.
  2. Podziel zbiór na część treningową (80%) i testową (20%) z ziarnem losowości `random_state=42`.
  3. Zainicjalizuj model `RandomForestClassifier`.
  4. Wytrenuj model na danych treningowych (`fit(X_train, y_train)`).
  5. Wygeneruj predykcje dla zbioru testowego (`predict(X_test)`).
  6. Oblicz `accuracy_score` porównując predykcje z faktycznymi etykietami `y_test`.
* **Oczekiwany rezultat:** Wartość `accuracy_score` jest większa lub równa 0.85.

**TC 1.2: Weryfikacja struktury tablicy predykcji**
* **Cel:** Upewnienie się, że model zwraca odpowiedni format danych po klasyfikacji.
* **Kroki testowe:**
  1. Wykonaj kroki 1-5 z TC 1.1.
  2. Sprawdź długość zwróconej tablicy predykcji.
  3. Sprawdź unikalne wartości w tablicy predykcji.
* **Oczekiwany rezultat:** Długość tablicy predykcji jest równa długości `y_test`. Tablica zawiera wyłącznie wartości 0 oraz 1.

---

## Scenariusz 2: Budowa i walidacja potoku przetwarzania (Pipeline)

**Cel testu:** Potwierdzenie prawidłowego działania klasy `Pipeline`, która stanowi kluczowy mechanizm zapobiegania wyciekom danych (data leakage) podczas wieloetapowej transformacji i uczenia.
* **Oczekiwany rezultat:** Zdefiniowany potok pomyślnie i w sposób ukryty dla użytkownika przetworzy surowe dane wejściowe. Transformator poprawnie przeskaluje dane wewnętrznie, przekazując je do klasyfikatora.
* **Kryterium zaliczenia:** Wywołanie metody `pipeline.predict()` na nieskalowanych danych testowych wykonuje się bez błędów formatowania oraz zwraca tablicę predykcji o długości odpowiadającej liczbie próbek.

### Przypadki testowe (Test Cases)

**TC 2.1: Przetwarzanie surowych danych testowych przez rurociąg**
* **Cel:** Weryfikacja, czy obiekt `Pipeline` automatycznie aplikuje transformacje z kroku pierwszego na dane z kroku drugiego.
* **Warunki wstępne:** Załadowany zbiór danych podzielony na `X_train` i `X_test`.
* **Kroki testowe:**
  1. Utwórz potok `Pipeline` składający się ze `StandardScaler` oraz modelu `SVC`.
  2. Wywołaj `pipeline.fit(X_train, y_train)`.
  3. Wywołaj `pipeline.predict(X_test)` (podając surowe, nieskalowane dane).
* **Oczekiwany rezultat:** Kod wykonuje się bez błędu niezgodności formatów, a metoda zwraca prawidłową tablicę przewidywań (skaler zadziałał w tle).

**TC 2.2: Weryfikacja braku wycieku danych (Data Leakage) w Pipeline**
* **Cel:** Potwierdzenie, że statystyki skalera nie nadpisują się podczas predykcji.
* **Kroki testowe:**
  1. Zainicjalizuj i wytrenuj `Pipeline` (jak w TC 2.1).
  2. Pobierz obiekt skalera z wytrenowanego potoku (`pipeline.named_steps['scaler']`).
  3. Zapisz wartość atrybutu `mean_` (wyliczona średnia).
  4. Wykonaj predykcję na nowym zbiorze testowym (`pipeline.predict(X_test)`).
  5. Ponownie sprawdź wartość atrybutu `mean_` skalera.
* **Oczekiwany rezultat:** Wartość `mean_` przed predykcją i po predykcji jest w 100% identyczna.

---

## Scenariusz 3: Zapisywanie i wczytywanie modelu (Persystencja)

**Cel testu:** Walidacja mechanizmu serializacji, upewniająca, że wyuczony model może zostać bezpiecznie zrzucony na dysk i bezstratnie odzyskany (Deployment).
* **Oczekiwany rezultat:** Przetrenowany model zostaje wyeksportowany do pliku binarnego, usunięty z pamięci, a następnie zainicjalizowany ponownie. Odtworzony obiekt zachowuje pełną strukturę i wszystkie wyuczone wagi.
* **Kryterium zaliczenia:** Odtworzony z dysku model generuje w 100% identyczny wektor prognoz dla wzorcowego zestawu danych testowych.

### Przypadki testowe (Test Cases)

**TC 3.1: Weryfikacja odtwarzalności predykcji po serializacji z joblib**
* **Cel:** Sprawdzenie absolutnej zgodności wag modelu przed i po zapisie na dysk.
* **Warunki wstępne:** Wytrenowany model klasyfikacyjny w pamięci, przygotowany zbiór testowy `X_test`.
* **Kroki testowe:**
  1. Użyj obecnego w pamięci modelu do wygenerowania przewidywań (`preds_original`).
  2. Zapisz model na dysk używając `joblib.dump(model, 'model.pkl')`.
  3. Usuń model z pamięci programu (`del model`).
  4. Załaduj model z pliku: `loaded_model = joblib.load('model.pkl')`.
  5. Wygeneruj predykcje odtworzonym modelem: `preds_loaded = loaded_model.predict(X_test)`.
  6. Porównaj zawartość `preds_original` z `preds_loaded`.
* **Oczekiwany rezultat:** Obie tablice są we wszystkich punktach identyczne (funkcja np. `np.array_equal` zwraca `True`), a operacja wczytania nie rzuca błędów.
