# Projekt zespołowy - System testowania OOB modułu scikit-learn

## Cel projektu
Celem projektu jest zaprojektowanie i zrealizowanie uproszczonego systemu testowania OOB (Out Of The Box) dla wybranego modułu Pythonowego. Zdecydowaliśmy się na przetestowanie biblioteki scikit-learn - popularnego narzędzia do klasycznego uczenia maszynowego.

Projekt obejmuje testowanie funkcjonalne i wydajnościowe, a także automatyzację procesu za pomocą manualnie uruchamianej pipeline w GitHub Actions. 

Aby dogłębnie sprawdzić stabilność biblioteki, nasz system zakłada pobranie wybranego commita z oficjalnego repozytorium GitHub modułu, zbudowanie go lokalnie i przetestowanie go przy użyciu naszej pipeline.

## Kanał komunikacji zespołu (TODO: dodac kiedy sie spotykamy, regularnie, poza tym moga byc zajecia)
Codzienną komunikację i podział pracy będziemy wykonywać na serwerze Discord.

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

## Harmonogram Projektu (ok. 2,5 miesiąca) TODO: TRZEBA DODAC ETA I DOD
Nasz projekt podzielony jest na etapy, zgodnie z punktami kontrolnymi:

- Punkt kontrolny 1 (Organizacja projektu): Założenie repozytorium, stworzenie pliku README z opisem, zdefiniowanie ról w zespole, ustalenie kanałów komunikacji i zaplanowanie wstępnych scenariuszy testowych.
- Punkt kontrolny 2 (Zarządzanie kodem): Rozpoczęcie pracy z Issues i Pull Requestami. Główny nacisk na wdrożenie Code Review oraz rozpoczęcie prac nad rozwiązywaniem zależności kompilacji scikit-learn ze źródeł.
- Punkt kontrolny 3 (Testowanie): Posiadanie w pełni działających testów funkcjonalnych i wydajnościowych. Pipeline uruchamia się poprawnie (buduje kod i odpala testy), a zespół posiada raportowanie wyników.
- Ocena końcowa (Release): Uzupełnienie kompletności dokumentacji, w tym udokumentowanie procesu budowania. Rzetelna samoocena zespołu, podział punktów i przygotowanie prezentacji z omówieniem napotkanych problemów.

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
