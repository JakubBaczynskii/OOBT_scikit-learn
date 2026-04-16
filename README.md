# Projekt zespołowy - System testowania OOB modułu scikit-learn

## Cel projektu
Celem projektu jest zaprojektowanie i zrealizowanie uproszczonego systemu testowania OOB (Out Of The Box) dla wybranego modułu Pythonowego. Zdecydowaliśmy się na przetestowanie biblioteki scikit-learn - popularnego narzędzia do klasycznego uczenia maszynowego.

Projekt obejmuje testowanie funkcjonalne i wydajnościowe, a także automatyzację procesu za pomocą manualnie uruchamianej pipeline w GitHub Actions. 

Aby dogłębnie sprawdzić stabilność biblioteki, nasz system zakłada pobranie wybranego commita z oficjalnego repozytorium GitHub modułu, zbudowanie go lokalnie i przetestowanie go przy użyciu naszej pipeline.

## Komunikacja i Zasady pracy z repozytorium
Bieżąca komunikacja i szybkie ustalenia odbywają się na dedykowanym serwerze Discord. Cotygodniowe spotkania odbywają się w czwartki o godzinie 17:30.

Aby utrzymać porządek w kodzie zespołu 4-osobowego, przyjęliśmy następujące zasady pracy technicznej na GitHubie:
- **Zarządzanie zadaniami (Issues):** Każde nowe zadanie, pomysł na test lub problem z buildem zgłaszamy najpierw jako GitHub Issue z odpowiednim tagiem (np. *bug*, *enhancement*, *documentation*).
- **Praca na gałęziach (Branching):** Nikt nie commituje bezpośrednio do głównej gałęzi `main`. Każdą nową funkcjonalność tworzymy na osobnym branchu (np. `feature/testy-wydajnosciowe`, `fix/build-error`).
- **Integracja kodu (Pull Requests):** Gotowy kod wprowadzamy do gałęzi `main` wyłącznie poprzez Pull Requesty (PR).
- **Code Review:** Żaden Pull Request nie może zostać zmergowany przez jego autora. Każdy kod wymaga sprawdzenia i zatwierdzenia (Approve) przez co najmniej jednego innego członka zespołu.
  
## Skład zespołu i podział ról
Ze względu na 4-osobowy skład zespołu i wysoki stopień skomplikowania procesu budowania modułu, podzieliliśmy zadania w następujący sposób:

1. [Adam Micun](https://github.com/Adamono) - Inżynier DevOps / CI/CD Architect
   - Odpowiedzialność: Konfiguracja repozytorium oraz stworzenie pipeline w GitHub Actions.
   - Główne wyzwanie: Zrozumienie zależności środowiskowych, rozwiązanie ewentualnych problemów buildowych (kompilatory C/C++, Cython) i zapewnienie poprawnego budowania modułu z kodu źródłowego (commita) przed uruchomieniem testów.

2. [Jakub Baczyński](https://github.com/JakubBaczynskii) - Inżynier Testów Funkcjonalnych (ML QA)
   - Odpowiedzialność: Zaprojektowanie i zaimplementowanie w kodzie od 3 do 5 testów funkcjonalnych. 
   - Główne wyzwanie: Sprawdzenie realnego użycia modułu (np. klasyfikacja, pipeline, metryki) wraz z odpowiednim uzasadnieniem i nazewnictwem.

3. [Emilia Wierzbanowska](https://github.com/emiliaw1) - Inżynier Wydajności (Performance QA)
   - Odpowiedzialność: Przygotowanie 1-2 prostych testów wydajnościowych.
   - Główne wyzwanie: Pomiar czasu wykonania wybranych operacji (np. trenowania modelu), zapis wyniku do loga oraz proste porównanie wyników w zależności od rozmiaru danych wejściowych.

4. [Mykola Mashovets](https://github.com/MykMash) - Analityk Testów / Technical Writer
   - Odpowiedzialność: Organizacja dokumentacji projektu oraz struktury katalogów.
   - Główne wyzwanie: Opracowanie dokumentu zawierającego co najmniej 3 scenariusze testów akceptacyjnych (cel, rezultat, kryterium zaliczenia) oraz szczegółowe udokumentowanie procesu budowania modułu ze źródeł.

## Harmonogram Projektu (ok. 2,5 miesiąca)
Nasz projekt podzielony jest na etapy, zgodnie z punktami kontrolnymi:

- **Tydzień 1-2 (do 17.04): Punkt kontrolny 1 (Organizacja projektu)** - Założenie repozytorium, stworzenie pliku README, zdefiniowanie ról, kanałów komunikacji.
- **Tydzień 3-5 (do 8.05): Punkt kontrolny 2 (Zarządzanie kodem)** - Praca z Issues, PR, wdrożenie Code Review i rozwiązywanie zależności kompilacji ze źródeł.
- **Tydzień 6-8 (do 29.05): Punkt kontrolny 3 (Testowanie)** - Działające testy funkcjonalne i wydajnościowe, poprawnie uruchamiająca się pipeline budująca kod ze źródeł, raportowanie wyników.
- **Tydzień 9-10 (do 12.06): Ocena końcowa (Release)** - Kompletna dokumentacja, podsumowanie problemów buildowych, prezentacja i rzetelna samoocena.

## Wstępne scenariusze testowe (Akceptacyjne)
Pełny opis scenariuszy znajduje się w dedykowanym dokumencie docs/scenariusze.md

1. Zdolność do klasyfikacji (Real Use-Case): Sprawdzenie, czy model RandomForestClassifier jest w stanie poprawnie przetrenować się na zbiorze Breast Cancer i wygenerować predykcje bez błędów.
2. Budowa Potoku (Pipeline): Weryfikacja bezbłędnego przepływu danych przez transformatory wstępne (np. StandardScaler) oraz estymator końcowy.
3. Persystencja (Zapis/Odczyt): Zbadanie, czy wyuczony model może zostać pomyślnie zapisany na dysk i załadowany ponownie zachowując identyczne wyniki predykcji.

## Struktura Katalogów
```text
.
├── .github/
│   └── workflows/
│       └── pipeline.yml       # Konfiguracja pipeline GitHub Actions
├── docs/
│   └── scenariusze.md         # Dokument z opisem testów akceptacyjnych
├── tests/
│   ├── functional/            # Skrypty testów funkcjonalnych (3-5 testów)
│   └── performance/           # Skrypty testów wydajnościowych (1-2 testy)
├── README.md                  # Główny plik informacyjny projektu
└── requirements.txt           # Zależności środowiskowe i buildowe
