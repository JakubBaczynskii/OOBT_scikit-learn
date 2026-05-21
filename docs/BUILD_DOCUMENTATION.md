# Dokumentacja Procesu Budowania i Code Review

---

## Przegląd Projektu

**Projekt:** System testowania OOB (Out Of The Box) dla scikit-learn  
**Cel:** Testowanie biblioteki bezpośrednio po zainstalowaniu, bez modyfikacji  
**Zespół:** 4 osoby (DevOps, ML QA, Performance QA, Technical Writer)  
**Stack:** Python 100% (scikit-learn, pytest, joblib)

---

## Architektura Testów

### **Testy Funkcjonalne** (3 testy zaimplementowane)

| Test | Plik | Cel | Scenariusz | Status |
|------|------|-----|-----------|--------|
| **Klasyfikacja** | `test_classification.py` | Zdolność do klasyfikacji | RandomForest na zbiorze Breast Cancer | DONE |
| **Pipeline** | `test_pipeline.py` | Budowa potoku przetwarzania | StandardScaler + SVC | DONE |
| **Persystencja** | `test_model_persistence.py` | Zapis/Odczyt modelu | LogisticRegression + joblib | DONE |

| **Clustering** | `test_clustering.py` | Grupowanie danych | KMeans na zbiorze | DONE |
| **Regresja** | `test_regression.py` | Modelowanie regresji | RandomForest + Linear Regression + Baseline | DONE |
| **Edge Cases** | `test_edge_cases.py` | Testy graniczne | Puste dane, invalid inputs | DONE |
| **Pipeline Integration** | `test_pipeline_integration.py` | Integracja pełnego pipeline'a | Wieloetapowy workflow | DONE |

### **Testy Wydajnościowe** (5 testów zaimplementowanych)

| Test | Plik | Cel | Scenariusz | Status |
|------|------|-----|-----------|--------|
| **Performance Scaling** | `test_performance.py` | Skalowanie na rozmiarach danych | 1000 vs 50000 próbek | DONE |
| **Performance Large Dataset** | `test_performance.py` | Wydajność na dużych danych | 50000 próbek, 100 estimators | DONE |
| **Performance Scaling Factor** | `test_performance.py` | Porównanie czasu scaling | Wzrost czasu z rozmiarem danych | DONE |
| **Parallelization** | `test_multiprocessing.py` | Testowanie wielowątkowości | n_jobs=1 vs n_jobs=-1 | DONE |
| **Sparse Performance** | `test_sparse_performance.py` | Wydajność na macierzach rzadkich | SGDClassifier na sparse matrices | DONE |

**Odpowiedzialność:** [Emilia Wierzbanowska](https://github.com/emiliaw1) (Performance QA)

---

## Wymagania Systemowe do Budowania

### **System Dependencies**

Aby budować scikit-learn ze źródeł, wymagane są:

```bash
# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
  build-essential \           # GCC, G++, Make
  gfortran \                  # Kompilator Fortran (dla LAPACK)
  libopenblas-dev \           # Biblioteka algebry liniowej
  liblapack-dev \             # Biblioteka LAPACK
```

**Dlaczego te zależności**
- **build-essential:** GCC i G++ wymagane do kompilacji rozszerzeń C
- **gfortran:** Wymagany do kompilacji kodu Fortran w LAPACK
- **libopenblas-dev:** Biblioteka do operacji na macierzach (linear algebra)
- **liblapack-dev:** Zaawansowane operacje algebry liniowej

### **Python Dependencies - Runtime**

Z `requirements.txt`:
```bash
scikit-learn                   # Główna biblioteka
pytest                         # Framework do testów
```

### **Python Dependencies - Build Tools**

```bash
pip install numpy scipy cython meson-python meson ninja setuptools wheel pytest pytest-cov
```

| Pakiet | Wersja | Cel |
|--------|--------|-----|
| **numpy** | LATEST | Bazowa biblioteka do tablic (wymagane przez scikit-learn) |
| **scipy** | LATEST | Naukowe funkcje (zależność scikit-learn) |
| **cython** | LATEST | Kompilator Python→C (wymagany do budowania) |
| **meson-python** | LATEST | Build system dla scikit-learn |
| **meson** | LATEST | Build koordynator |
| **ninja** | LATEST | Szybki build tool |
| **setuptools** | LATEST | Narzędzie pakowania |
| **wheel** | LATEST | Format pakietów binarnych |
| **pytest** | LATEST | Framework do testów |
| **pytest-cov** | LATEST | Coverage plugin dla pytest (nowy) |

---

## Zależności Testów - Szczegółowa Analiza

### **Test 1: Klasyfikacja (`tests/functional/test_classification.py`)**

```python
from sklearn.datasets import load_breast_cancer          # Zbiór danych
from sklearn.ensemble import RandomForestClassifier      # Model
from sklearn.model_selection import train_test_split     # Podział danych
from sklearn.metrics import accuracy_score               # Metryka
```

| Zależność | Typ | Cel | Wymagana? | Notatka |
|-----------|-----|-----|----------|---------|
| `load_breast_cancer` | Zbiór danych | Dane treningowe (569 próbek) | YES | Wbudowany w scikit-learn |
| `RandomForestClassifier` | Model | Klasyfikator ensemble | YES | Wymaga kompilacji C/Cython |
| `train_test_split` | Narzędzie | Podział 80/20 danych | YES | Pure Python |
| `accuracy_score` | Metryka | Ocena dokładności | YES | Pure Python |

**Wymagania do budowania:**
- ✅ Python 3.11+ (scikit-learn 1.4+)
- ✅ Kompilatory C/C++ (RandomForest ma OptimizedTree w C++)
- ✅ NumPy (zależność scikit-learn)
- ✅ SciPy (zależność scikit-learn)
- ✅ Joblib (zależność scikit-learn)

---

### **Test 2: Pipeline (`tests/functional/test_pipeline.py`)**

```python
from sklearn.pipeline import Pipeline                     # Koordynator
from sklearn.preprocessing import StandardScaler          # Transformator
from sklearn.svm import SVC                              # Klasyfikator
from sklearn.datasets import make_classification          # Syntetyczne dane
```

| Zależność | Typ | Cel | Wymagana? | Notatka |
|-----------|-----|-----|----------|---------|
| `Pipeline` | Klasa | Łączy transformatory + model | YES | Pure Python |
| `StandardScaler` | Transformator | Normalizuje dane | YES | Używa NumPy |
| `SVC` | Model | Support Vector Classifier | YES | **WYMAGA KOMPILACJI - Libsvm** |
| `make_classification` | Generator | Syntetyczne dane | YES | Pure Python |

**Wymagania do budowania:**
- ✅ Kompilator C++ (SVC + Libsvm library)
- ✅ BLAS/LAPACK (operacje na macierzach)
- ✅ Cython (kompilacja rozszerzeń)
- ⚠️ **KRYTYCZNE:** SVC jest najbardziej wymagającym modelem w scikit-learn

---

### **Test 3: Trwałość (`test_model_persistence.py`)**

```python
import joblib                                              # Serializacja
from sklearn.linear_model import LogisticRegression       # Model
from sklearn.datasets import load_iris                    # Zbiór danych
```

| Zależność | Typ | Cel | Wymagana? | Notatka |
|-----------|-----|-----|----------|---------|
| `joblib` | Biblioteka | Zapis/odczyt modelu | YES | Zależność scikit-learn |
| `LogisticRegression` | Model | Klasyfikator liniowy | YES | Mniej kompilacji niż SVC |
| `load_iris` | Zbiór danych | 150 próbek, 4 cechy | YES | Wbudowany w scikit-learn |

**Wymagania do budowania:**
- ✅ Standard library Python (`os`)
- ✅ Joblib (zależność scikit-learn)
- ✅ Mniej wymagający niż inne testy

---

## Analiza GitHub Actions Workflow

**Plik:** `.github/workflows/OOBT_scikit-learn_workflow.yml`

### **Aktywacja - Kiedy workflow się uruchamia:**

```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
```

**Trigger:** 
- Automatycznie przy push/PR na gałąź `main`
- Ręczne uruchamianie (workflow_dispatch)

---

## Szczegółowa Analiza Każdego Kroku

### **Krok 1: Checkout Repository (Pobranie repozytorium)**

```yaml
- name: Checkout repository
  uses: actions/checkout@v4
```

| Element | Opis |
|---------|------|
| `name:` | Nazwa kroku wyświetlana w GitHub UI |
| `uses: actions/checkout@v4` | Oficjalna GitHub akcja do pobrania kodu |
| `@v4` | Wersja akcji (v4 - stabilna, rekomendowana) |

---

### **Krok 2: Set up Python (Konfiguracja Python - ZMIENIONY KROK!)**

```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
```

| Element | Opis |
|---------|------|
| `uses: actions/setup-python@v5` | Oficjalna GitHub akcja do ustawienia Python |
| `@v5` | Wersja akcji |
| `with:` | Parametry konfiguracji |
| `python-version: ${{ matrix.python-version }}` | Wersja Python z matrix strategy |

**Nowe: Matrix Testing Strategy** (ZMIANA!)

```yaml
strategy:
  fail-fast: false
  matrix:
    python-version: ["3.11", "3.12", "3.13", "3.14"]
```

**Zalecenie:** Workflow testuje na **4 wersjach Pythona** jednocześnie:
- ✅ Python 3.11
- ✅ Python 3.12
- ✅ Python 3.13
- ✅ Python 3.14 (obsługiwany w najnowszych wersjach setup-python)

**Znaczenie:** 
- Gwarantuje kompatybilność na różnych wersjach
- `fail-fast: false` oznacza że wszystkie wersje są testowane niezależnie
- Jeśli jedna wersja zawiedzie, inne dalej się testują

---

### **Krok 3: Environment Variables (NOWY KROK!)**

```yaml
env:
  OS: ubuntu-24.04
  ARCH: x86_64
  OPENBLAS_VERSION: libopenblas-dev
  LAPACK_VERSION: liblapack-dev
```

**Nowe: Zmienne środowiskowe** (ZMIANA!)

| Zmienna | Wartość | Cel |
|---------|---------|-----|
| `OS` | `ubuntu-24.04` | Identyfikacja systemu operacyjnego |
| `ARCH` | `x86_64` | Architektura procesora |
| `OPENBLAS_VERSION` | `libopenblas-dev` | Wersja biblioteki OpenBLAS |
| `LAPACK_VERSION` | `liblapack-dev` | Wersja biblioteki LAPACK |

**Znaczenie:** Pozwala na łatwe aktualizowanie wersji zależności w jednym miejscu.

---

### **Krok 4: Display environment information (NOWY KROK!)**

```yaml
- name: Display environment information
  run: |
    echo "# Environment Information" > environment.md
    echo "" >> environment.md
    echo "- OS: Ubuntu 24.04" >> environment.md
    echo "- Architecture: $(uname -m)" >> environment.md
    echo "- Python version: $(python --version)" >> environment.md
    echo "- GCC version: $(gcc --version | head -n 1)" >> environment.md
    echo "- GFortran version: $(gfortran --version | head -n 1)" >> environment.md
```

**Nowy krok generuje plik `environment.md`** z informacjami o środowisku:
- OS
- Architektura
- Wersja Pythona
- Wersja GCC
- Wersja GFortrana

**Cel:** Ułatwia debugging - każdy run ma dokumentację środowiska.

---

### **Krok 5: Install system dependencies (Instalacja zależności systemowych)**

```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev
```

| Linia | Opis |
|-------|------|
| `run:` | Bezpośrednio uruchom shell command |
| `\|` | Multi-line command (pozwala na wiele linii) |
| `sudo apt-get update` | Zaktualizuj listę pakietów |
| `sudo apt-get install -y` | Zainstaluj pakiety (bez pytania -y) |
| `build-essential` | Kompilatory C/C++ (gcc, g++, make) |
| `gfortran` | Kompilator Fortran (dla LAPACK) |
| `libopenblas-dev` | Biblioteka BLAS (operacje na macierzach) |
| `liblapack-dev` | Biblioteka LAPACK (zaawansowana algebra liniowa) |

**Dlaczego to jest ważne:**
- `RandomFores` w C++ wymaga kompilatora
- `SVC (Libsvm)` wymaga kompilatora C++
- `LAPACK` jest kluczowy do operacji macierzowych
- Bez tego build failuje z błędami kompilacji

---

### **Krok 6: Install build dependencies (Instalacja build tools - ZMIENIONY!)**

```yaml
- name: Install build dependencies 
  run: | 
    python -m pip install --upgrade pip 
    pip install numpy scipy cython meson-python meson ninja setuptools wheel pytest pytest-cov
```

| Linia | Opis |
|-------|------|
| `python -m pip install --upgrade pip` | Upgrade pip (package manager) |
| `pip install` | Zainstaluj pakiety z PyPI |
| `numpy` | Bazowe tablice numeryczne |
| `scipy` | Naukowe funkcje (zależność scikit-learn) |
| `cython` | Kompilator Python→C |
| `meson-python` | Build system frontend dla scikit-learn |
| `meson` | Build orchestrator |
| `ninja` | Szybki build backend |
| `setuptools` | Tradycyjne narzędzie pakowania |
| `wheel` | Format pakietów binarnych |
| `pytest` | Framework do uruchamiania testów |
| `pytest-cov` | **NOWY** - Coverage reporting dla pytest |

**Nowy pakiet: pytest-cov** (ZMIANA!)
- Generuje raporty pokrycia kodu (coverage reports)
- Integruje się z pytest
- Pozwala na HTML reports (htmlcov/)

**Dlaczego to jest ważne:**
- `pip upgrade` jest wymagane do poprawnej instalacji
- `numpy` - zależność dla scikit-learn
- `scipy` - zależność dla scikit-learn
- `cython` jest wymagany do budowania scikit-learn
- `meson-python` - system budowania scikit-learn 1.4+
- `meson` jest wymagany przez meson-python
- `ninja` jest wymagany przez meson
- `setuptools` - legacy wsparcie
- `wheel` - format binarny
- `pytest` - uruchamianie testów
- `pytest-cov` - pokrycie testów (nowy)

**Znaczenie:** To są kluczowe narzędzia do budowania scikit-learn ze źródeł.

---

### **Krok 7: Build scikit-learn from source (Budowanie ze źródeł)**

```yaml
- name: Build scikit-learn from source
  run: |
    git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git
    cd scikit-learn

    pip install \
      --verbose \
      --no-build-isolation \
      --editable .
```

| Linia | Opis |
|-------|------|
| `git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git` | Pobierz ostatni commit scikit-learn |
| `--depth 1` | Tylko ostatni commit (performance) |
| `cd scikit-learn` | Przejdź do katalogu scikit-learn |
| `pip install --verbose --no-build-isolation --editable .` | Zainstaluj w dev mode |
| `--verbose` | Wyświetlaj szczegółowy output (debugging) |
| `--no-build-isolation` | Buduj bezpośrednio w tym środowisku |
| `--editable` | Dev mode (zmiany widoczne od razu) |
| `.` | W bieżącym katalogu |

**Co się dzieje:**
1. Pobiera ostatni kod `scikit-learn` z GitHub
2. Przechodzi do katalogu `scikit-learn`
3. Buduje bibliotekę ze źródeł
4. Kompiluje rozszerzenia C/C++/Cython
5. Instaluje w trybie `editable`

**Potencjalne błędy na tym kroku:**
```
error: could not compile C extension
error: gfortran: command not found
error: fatal: 'numpy/arrayobject.h' file not found
ImportError: libopenblas.so.0: cannot open shared object
```

**Rozwiązania:**
- ✅ System dependencies (Krok 5) - już dodane!
- ✅ Build dependencies (Krok 6) - już dodane!
- Sprawdzić połączenie (clone z GitHub)

---

### **Krok 8: Run tests and save report in Markdown (ZMIENIONY KROK!)**

```yaml
- name: Run tests and save report in Markdown
  run: |
    mkdir -p reports
    
    echo "# Test Results" > reports/test_report_final.md
    echo "" >> reports/test_report_final.md
    
    echo "## Executed tests" >> reports/test_report_final.md
    echo "" >> reports/test_report_final.md
    
    # Uruchomienie testów z pełnym verbose
    pytest tests/ \
      --cov=sklearn \
      --cov-report=html \
      --disable-warnings \
      -v \
      -rA \
      --maxfail=0 \
      2>&1 | tee reports/test_report.log
    
    echo "" >> reports/test_report_final.md
    echo "## Detailed pytest output" >> reports/test_report_final.md
    echo "" >> reports/test_report_final.md
    echo '```' >> reports/test_report_final.md
    
    cat reports/test_report.log >> reports/test_report_final.md
    
    echo '```' >> reports/test_report_final.md
    
    echo "" >> reports/test_report_final.md
    echo "## Collected tests" >> reports/test_report_final.md
    echo "" >> reports/test_report_final.md
    
    pytest --collect-only -q tests/ >> reports/test_report_final.md
```

**ZNACZĄCE ZMIANY W TEŚCIE:**

| Stary kod | Nowy kod | Zmiana |
|-----------|----------|--------|
| `pytest tests/functional --tb=short` | `pytest tests/ --cov=sklearn --cov-report=html` | **Coverage reporting** |
| `> test_results.txt` | `2>&1 \| tee reports/test_report.log` | **Lepsze logowanie** |
| `pytest tests/performance` | `--disable-warnings -v -rA` | **Pełna diagnostyka** |
| `\|\| true` | Brak (workflow zawiedzie) | **Strict mode** |

**Nowe parametry pytest:**

| Parametr | Cel |
|----------|-----|
| `--cov=sklearn` | Mierzy pokrycie kodu dla modułu sklearn |
| `--cov-report=html` | Generuje HTML report pokrycia |
| `--disable-warnings` | Ukrywa ostrzeżenia (czytelnie) |
| `-v` | Verbose mode (szczegóły każdego testu) |
| `-rA` | Report all (summary wszystkich testów) |
| `--maxfail=0` | Uruchomij wszystkie testy (nie zatrzymuj się na błędzie) |
| `2>&1 \| tee` | Zapisz output do pliku i wyświetl na ekranie |

**Wyniki:**
- ✅ `reports/test_report.log` - raw output testów
- ✅ `reports/test_report_final.md` - sformatowany raport Markdown
- ✅ `htmlcov/` - HTML coverage report

---

### **Krok 9: Display test results in summary (ZMIENIONY KROK!)**

```yaml
- name: Display test results in summary
  if: always()
  run: |
    echo "## Test Results" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    
    if [ -f reports/test_report.log ]; then
      echo '```' >> $GITHUB_STEP_SUMMARY
    
      # pokaż ostatnie 100 linii aby było widać wykonane testy
      tail -n 100 reports/test_report.log >> $GITHUB_STEP_SUMMARY
    
      echo '```' >> $GITHUB_STEP_SUMMARY
    else
      echo "No test report found" >> $GITHUB_STEP_SUMMARY
    fi
```

**Zmiana:** Wyświetla ostatnie 100 linii (zamiast 20) + komentarz wyjaśniający

---

### **Krok 10: Generate markdown summary (NOWY KROK!)**

```yaml
- name: Generate markdown summary
  if: always()
  run: |
    {
      echo "# CI Test Summary"
      echo ""
      echo "## Environment"
      echo ""
      cat environment.md
      echo ""
      echo "## Last 30 lines of test output"
      echo ""
      echo '```'
      
      if [ -f reports/test_report_final.md ]; then
        tail -n 30 reports/test_report_final.md
      else
        echo "No test report generated"
      fi
      
      echo '```'
    } >> $GITHUB_STEP_SUMMARY
```

**Nowy krok łączy wszystko w jednym podsumowaniu:**
- Informacje o środowisku
- Ostatnie linie testów
- Sformatowany output

---

### **Krok 11: Upload test report (per-version - NOWY KROK!)**

```yaml
- name: Upload test report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: test_report_final-py${{ matrix.python-version }}
    path: reports/test_report_final.md
    retention-days: 7
```

**Nowy krok:** Upload per-Python-version
- Nazwa artefaktu zawiera wersję Pythona
- Pozwala porównywać wyniki między wersjami

---

### **Krok 12: Upload coverage report (NOWY KROK!)**

```yaml
- name: Upload coverage report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: coverage-report-py${{ matrix.python-version }}
    path: htmlcov/
    retention-days: 7
```

**Nowy krok:** Upload HTML coverage report
- Zawiera dokładną analizę pokrycia kodu
- Można przeglądać w przeglądarce

---

### **Krok 13: Upload environment information (NOWY KROK!)**

```yaml
- name: Upload environment information
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: environment-py${{ matrix.python-version }}
    path: environment.md
    retention-days: 7
```

**Nowy krok:** Upload informacji o środowisku
- Zawiera wersje kompilatora, OS, etc.
- Ułatwia debugging problemów środowiskowych

---

## Podsumowanie Zmian

### **DODANE (NEW)**
- ✅ Matrix testing na 4 wersjach Pythona
- ✅ Environment variables (OS, ARCH, OPENBLAS_VERSION, LAPACK_VERSION)
- ✅ Krok Display environment information
- ✅ pytest-cov dla coverage reporting
- ✅ Coverage HTML reports (htmlcov/)
- ✅ Markdown test reports
- ✅ Per-version artefakty
- ✅ Comprehensive summary generation

### **ZMIENIONE (MODIFIED)**
- 🔄 Workflow z `python-version: '3.12'` na `matrix.python-version`
- 🔄 Build dependencies z dodanym `pytest-cov`
- 🔄 Test execution z `--cov` i `--cov-report=html`
- 🔄 Usunięte `|| true` (strict mode)
- 🔄 Ulepszone logowanie (tee + multiple reports)

### **USUNIĘTE (REMOVED)**
- ❌ `|| true` przy testach (teraz workflow zawiedzie jeśli testy zawiodą)
- ❌ Proste `test_results.txt` (zastąpione strukturyzowanymi raportami)
- ❌ Jeden upload artefaktu (teraz 3+ uploady per version)

---

## Napotkane Problemy i Rozwiązania

### **Problem 1: Python 3.14 nie istnieje** FIXED

```diff
- python-version: '3.14'
+ strategy:
+   matrix:
+     python-version: ["3.11", "3.12", "3.13", "3.14"]
```

**Stan:** FIXED - matrix testing na wszystkich wersjach

---

### **Problem 2: Brakujące zależności systemowe** FIXED

**Objawy (z poprzedniej wersji):**
```
error: could not compile C extension module for sklearn
error during compilation of scikit-learn
ImportError: DLL load failed - SVM not found
```

**Rozwiązanie (teraz zaimplementowane):**

```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev
```

**Status:** FIXED w Kroku 5

**Zainstalowane pakiety:**
- ✅ `build-essential` - GCC, G++, Make
- ✅ `gfortran` - Kompilator Fortran
- ✅ `libopenblas-dev` - Biblioteka BLAS
- ✅ `liblapack-dev` - Biblioteka LAPACK

---

### **Problem 3: Brakujące build tools do kompilacji** FIXED

**Objawy (z poprzedniej wersji):**
```
error: Cython not found
error: meson build system not found
ImportError: No module named 'setuptools'
```

**Rozwiązanie (teraz zaimplementowane):**

```yaml
- name: Install build dependencies
  run: |
    python -m pip install --upgrade pip 
    pip install numpy scipy cython meson-python meson ninja setuptools wheel pytest pytest-cov
```

**Status:** FIXED w Kroku 6

**Zainstalowane pakiety:**
- ✅ `numpy` - Bazowe tablice
- ✅ `scipy` - Naukowe funkcje
- ✅ `cython` - Kompilator Python→C
- ✅ `meson-python` - Build system scikit-learn
- ✅ `meson` - Build orchestrator
- ✅ `ninja` - Szybki build tool
- ✅ `setuptools` - Tradycyjne narzędzie
- ✅ `wheel` - Format pakietów
- ✅ `pytest` - Framework testów
- ✅ `pytest-cov` - Coverage reporting (nowy)

---

### **Problem 4: Brak budowania ze źródeł** FIXED

**Objawy (z poprzedniej wersji):**
```
ImportError: sklearn not found (pre-built wheels only)
Brak kompilacji rozszerzeń C/C++
```

**Rozwiązanie (teraz zaimplementowane):**

```yaml
- name: Build scikit-learn from source
  run: |
    git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git
    cd scikit-learn
    pip install --verbose --no-build-isolation --editable .
```

**Status:** FIXED w Kroku 7

**Co to robi:**
- ✅ Pobiera najnowszy kod scikit-learn
- ✅ Buduje rozszerzenia C/C++/Cython
- ✅ Kompiluje LAPACK/BLAS operacje
- ✅ Instaluje w dev mode

---

### **Problem 5: Brakujące testy wydajnościowe** FIXED

**Plik:** `tests/performance/` - **ZAWIERA 5 TESTÓW**

**Odpowiedzialność:** [Emilia Wierzbanowska](https://github.com/emiliaw1) (Performance QA)  

**Stan:** IMPLEMENTACJA ZAKOŃCZONA

**Zaimplementowane testy:**

1. **`test_performance.py`** (3 testy):
   - `test_fit_performance_small_dataset()` - Trening na 1000 próbkach
   - `test_fit_performance_large_dataset()` - Trening na 50000 próbkach
   - `test_performance_scaling()` - Porównanie scaling factor

2. **`test_multiprocessing.py`** (1 test):
   - `test_random_forest_parallelization()` - n_jobs=1 vs n_jobs=-1, duży zbiór 100K próbek

3. **`test_sparse_performance.py`** (1 test):
   - `test_sparse_performance()` - Dense vs Sparse matrices, SGDClassifier

**Metryki mierzone:**
- ✅ Czas treningu (time.perf_counter(), time.time())
- ✅ Przyspieszenie równoległe (speedup factor)
- ✅ Wydajność na macierzach rzadkich

**Integracja z workflow:**
- ✅ Testy uruchamiają się w Kroku 8 (pytest tests/)
- ✅ Wyniki logują się do reports/test_report.log
- ✅ Coverage mierzy się dla wszystkich testów
- ✅ Artefakty uploadują się do GitHub (Kroki 11-13)

---

### **Problem 6: Brak widoczności testów** FIXED

**Objawy (z poprzedniej wersji):**
```
Testy się uruchomiły ale nie wiadomo ile i których
Brak raportów w GitHub UI
```

**Rozwiązanie (teraz zaimplementowane):**
- ✅ Markdown reports (`test_report_final.md`)
- ✅ Coverage reports (HTML)
- ✅ GitHub summary z 100 liniami output
- ✅ Environment metadata
- ✅ Multiple artefakty per version

**Status:** FIXED - kompleksowe raportowanie

---

### **Problem 7: Brak pokrycia kodu** FIXED

**Objawy (z poprzedniej wersji):**
```
Nie wiadomo jak dobre są testy
Brak metryki coverage
```

**Rozwiązanie (teraz zaimplementowane):**
```yaml
pytest tests/ \
  --cov=sklearn \
  --cov-report=html
```

**Status:** FIXED - pytest-cov integracja

**Wyniki:**
- ✅ `htmlcov/index.html` - interaktywny report
- ✅ Coverage % dla każdego modułu
- ✅ Line-by-line coverage visualization

---

### **Problem 8: Workflow zawsze się powiedzie** FIXED

**Objawy (z poprzedniej wersji):**
```
Testy mogą zawiać ale workflow shows ✅
PR merguje się mimo błędów testów
```

**Rozwiązanie (teraz zaimplementowane):**

```diff
- pytest tests/functional --tb=short --verbose > test_results.txt 2>&1 || true
- pytest tests/performance --tb=short --verbose >> test_results.txt 2>&1 || true
+ pytest tests/ --cov=sklearn --cov-report=html --disable-warnings -v -rA --maxfail=0 2>&1 | tee reports/test_report.log
```

**Usunięto `|| true`** - workflow zawiedzie jeśli testy zawiodą

**Status:** FIXED - strict mode enabled

---

## Checklist Code Review

Gdy inżynierowie QA będą commitować PR-y z testami, sprawdzać:

### **Aspekt: Build & Zależności**
- [ ] Czy workflow się pomyślnie uruchamia na **wszystkich 4 wersjach Pythona**?
- [ ] Czy build ze źródeł przechodzi na każdej wersji?
- [ ] Czy wszystkie kroki 1-7 działają bez błędów?
- [ ] Czy coverage report jest generowany?
- [ ] Czy environment.md zawiera poprawne dane?

### **Aspekt: Testowanie**
- [ ] Czy zarówno functional i performance testy się uruchamiają?
- [ ] Czy `test_report_final.md` jest generowany?
- [ ] Czy wyniki są widoczne w GitHub Summary (Krok 9)?
- [ ] Czy artefakty są uploadowane (Krok 11-13)?
- [ ] Czy coverage report pokazuje znaczące pokrycie?

### **Aspekt: Jakość testów**
- [ ] Czy test ma jasne sekcje (Arrange-Act-Assert)?
- [ ] Czy ma docstring wyjaśniający co testuje?
- [ ] Czy assertion ma wiadomość o błędzie?
- [ ] Czy test jest niezależny od innych testów?
- [ ] Czy działa na wszystkich 4 wersjach Pythona?

### **Aspekt: Code Review**
- [ ] Czy commit message jest jasny i opisowy?
  - ✅ Dobry: `"feat: Add performance test for RandomForest training time"`
  - ❌ Zły: `"fix"`, `"update"`, `"test"`
- [ ] Czy PR description wyjaśnia co zmienia?
- [ ] Czy są testy dla nowego kodu?
- [ ] Czy kod jest sformatowany (PEP8)?
- [ ] Czy testuje się na wszystkich wersjach Pythona?

### **Aspekt: Dokumentacja**
- [ ] Czy test jest udokumentowany?
- [ ] Czy jest opisane co się testuje i dlaczego?
- [ ] Czy są skomentowane złożone sekcje?
- [ ] Czy scenariusz testowy jest jasny?
- [ ] Czy coverage jest wystarczający?

---

## Przeznaczenie Każdego Pliku Testów

### **Testy Funkcjonalne**

| Plik | Autor | Cel | Wymagania | Status |
|------|-------|-----|----------|--------|
| `test_classification.py` | [Jakub](https://github.com/JakubBaczynskii) (ML QA) | Rzeczywisty use-case klasyfikacji | sklearn datasets, RandomForest | READY |
| `test_pipeline.py` | [Jakub](https://github.com/JakubBaczynskii) (ML QA) | Weryfikacja klasy Pipeline | sklearn preprocessing, SVC | READY |
| `test_model_persistence.py` | [Jakub](https://github.com/JakubBaczynskii) (ML QA) | Zapis/odczyt modelu | joblib, LogisticRegression | READY |
| `test_clustering.py` | [Jakub](https://github.com/JakubBaczynskii) (ML QA) | Testowanie KMeans | KMeans clustering | READY |
| `test_regression.py` | [Jakub](https://github.com/JakubBaczynskii) (ML QA) | Regresja + baseline | RandomForest, Linear, Dummy | READY |
| `test_edge_cases.py` | [Jakub](https://github.com/JakubBaczynskii) (ML QA) | Testy graniczne | Invalid inputs, edge cases | READY |
| `test_pipeline_integration.py` | [Jakub](https://github.com/JakubBaczynskii) (ML QA) | Integracja end-to-end | Multi-step pipeline | READY |

### **Testy Wydajnościowe**

| Plik | Autor | Cel | Metryki | Status |
|------|-------|-----|---------|--------|
| `test_performance.py` | [Emilia](https://github.com/emiliaw1) (Performance QA) | Scaling performance | Training time, scaling factor | DONE |
| `test_multiprocessing.py` | [Emilia](https://github.com/emiliaw1) (Performance QA) | Parallelization speedup | n_jobs effect, speedup factor | DONE |
| `test_sparse_performance.py` | [Emilia](https://github.com/emiliaw1) (Performance QA) | Dense vs Sparse | Training time comparison | DONE |

---

## Struktura Katalogów (Docelowa)

```
OOBT_scikit-learn/
├── .github/
│   └── workflows/
│       └── OOBT_scikit-learn_workflow.yml      # ✅ Zaktualizowany workflow
├── docs/
│   ├── scenariusze.md                          # Scenariusze testów akceptacyjnych
│   ├── BUILD_DOCUMENTATION.md                  # Ten plik (zaktualizowany)
├── tests/
│   ├── functional/
│   │   ├── test_classification.py              # READY
│   │   ├── test_pipeline.py                    # READY
│   │   ├── test_model_persistence.py           # READY
│   │   ├── test_clustering.py                  # READY
│   │   ├── test_regression.py                  # READY
│   │   ├── test_edge_cases.py                  # READY
│   │   └── test_pipeline_integration.py        # READY
│   └── performance/
│       ├── test_performance.py                 # DONE
│       ├── test_multiprocessing.py             # DONE
│       └── test_sparse_performance.py          # DONE
├── reports/
│   ├── test_report_final.md                    # Sformatowany Markdown report
│   └── test_report.log                         # Raw pytest output
├── htmlcov/                                    # Coverage HTML report (generated)
├── README.md                                   # Główna dokumentacja
├── requirements.txt                            # Zależności (scikit-learn, pytest)
└── .gitignore
```

---

## Quick Start - Uruchomienie Testów Lokalnie

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/JakubBaczynskii/OOBT_scikit-learn.git
cd OOBT_scikit-learn

# 2. Zainstaluj zależności systemowe (Linux)
sudo apt-get update
sudo apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev

# 3. Stwórz wirtualne środowisko
python3 -m venv sklearn_env
source sklearn_env/bin/activate

# 4. Zainstaluj build tools (WITH COVERAGE)
pip install --upgrade pip
pip install numpy scipy cython meson-python meson ninja setuptools wheel pytest pytest-cov

# 5. Sbuduj scikit-learn ze źródeł
git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
pip install --verbose --no-build-isolation --editable .
cd ..

# 6. Uruchom wszystkie testy z coverage
mkdir -p reports
pytest tests/ \
  --cov=sklearn \
  --cov-report=html \
  --disable-warnings \
  -v \
  -rA \
  --maxfail=0 \
  2>&1 | tee reports/test_report.log

# 7. Uruchom konkretny test
pytest tests/functional/test_classification.py -v

# 8. Pokaż HTML coverage report
open htmlcov/index.html  # macOS
# lub
xdg-open htmlcov/index.html  # Linux

# 9. Pokaż wyniki
cat reports/test_report.log
```

---

## Zadania dla [Mykoli](https://github.com/MykMash) (Technical Writer)

### **Task 1.1: Monitoring Pipeline DevOps** ✅
- [x] Obserwować workflow przy każdym push/PR
- [x] Zapisywać jakie błędy się pojawiają
- [x] Notować czasy budowania
- [x] Dokument zaktualizowany - wszystkie kroki udokumentowane

### **Task 1.2: Dokumentacja Procesu Budowania** ✅
- [x] Przygotować listę zależności systemowych
- [x] Dokumentować każdy krok budowania (14 kroków)
- [x] Wyjaśnić zależności dla każdego testu
- [x] Stwórzyć Quick Start
- [x] Udokumentować zmiany w workflowie (matrix testing, coverage, etc.)

### **Task 1.3: Standardy Code Review** ✅
- [x] Stwórzyć checklist dla recenzentów
- [x] Dokumentować wymagania dla testów na 4 wersjach Pythona
- [x] Ustandaryzować commit messages
- [x] Dokumentacja zmian i ich znaczenia

### **Task 1.4: Dokumentacja Błędów Budowania** ✅
- [x] Udokumentować wszystkie naprawione problemy
- [x] Dodać Problem 8 (workflow zawsze się powiedzie - FIXED)
- [x] Oznaczyć Status dla każdego problemu

### **Task 1.5: Aktualizacja Testów Wydajnościowych** ✅
- [x] Zaktualizować tabelę testów funkcjonalnych (7 testów)
- [x] Zaktualizować tabelę testów wydajnościowych (5 testów)
- [x] Oznaczyć Problem 5 jako FIXED
- [x] Dodać szczegóły implementacji wydajności

---

## Status Problemów (Summary)

| # | Problem | Poprzedni Status | Obecny Status | Zmiana |
|---|---------|------------------|---------------|---------| 
| 1 | Python 3.14 | CRITICAL | FIXED | ✅ |
| 2 | System deps | CRITICAL | FIXED | ✅ |
| 3 | Build tools | CRITICAL | FIXED | ✅ |
| 4 | Build source | CRITICAL | FIXED | ✅ |
| 5 | Performance testy | AWAITING | FIXED  | ✅ |
| 6 | Timeout build | OPTIONAL | MONITORING | - |
| 7 | Brak pokrycia | CRITICAL | FIXED | ✅ |
| 8 | Workflow zawsze OK | CRITICAL | FIXED | ✅ |

---

## Kolejne Kroki

### **Milestone 2 (Teraz - Do 2026-05-21)** COMPLETE
1. ✅ Workflow zaktualizowany - matrix testing na 4 wersjach Pythona
2. ✅ Coverage reporting włączony
3. ✅ Dokumentacja pełna - 13 kroków szczegółowo opisanych
4. ✅ Strict mode - workflow zawiedzie jeśli testy zawiodą
5. ✅ Testy wydajnościowe zaimplementowane (5 testów)

### **Milestone 3 (2026-05-21 - 2026-05-29)** 🟢 IN PROGRESS
1. ✅ Emilia implementuje testy wydajnościowe - DONE
2. ⏳ Monitorować workflow - szukaj błędów na różnych wersjach Pythona
3. ⏳ Optymizować czas buildowania
4. ⏳ Obserwować ci-pipeline i dokumentować problemy

### **Milestone 4 (Release)**
1. ⏳ Finalizować dokumentację
2. ⏳ Zweryfikować wszystkie testy działają na Python 3.11, 3.12, 3.13, 3.14
3. ⏳ Przygotować release notes
4. ⏳ Archiwizować wyniki testów i coverage reports

---

## Kluczowe Elementy Bieżącego Workflow

### **Kroki 1-3: Przygotowanie**
- ✅ Checkout kodu (v4)
- ✅ Setup Python (matrix - 4 wersje)
- ✅ Environment variables + metadata collection

### **Kroki 4-6: Zależności**
- ✅ Display environment information
- ✅ System dependencies (complete)
- ✅ Build tools (complete + pytest-cov)

### **Krok 7: Budowanie**
- ✅ Build scikit-learn ze źródeł
- ✅ Kompilacja C/C++/Cython
- ✅ Meson build system

### **Krok 8: Testowanie (EXPANDED)**
- ✅ Functional tests (7 testów - [Jakub](https://github.com/JakubBaczynskii) - DONE)
- ✅ Performance tests (5 testów - [Emilia](https://github.com/emiliaw1) - DONE)
- ✅ Coverage reporting (htmlcov/)
- ✅ Structured markdown reports

### **Kroki 9-13: Raportowanie (EXPANDED)**
- ✅ Display wyników (GitHub UI - 100 linii)
- ✅ Generate markdown summary (comprehensive)
- ✅ Upload test reports (per-version)
- ✅ Upload coverage reports (per-version)
- ✅ Upload environment info (per-version)

---

**Dokument zaktualizowany:** 2026-05-21 (Update 2)  
**Ostatnia aktualizacja:** Aktualizacja tabel testów wydajnościowych i Problem 5  
**Status:** Complete - Testy wydajnościowe zaktualizowane i udokumentowane  
**Autor:** [Mykola Mashovets](https://github.com/MykMash)  
**Wersja:** 2.1 (Performance Tests Implementation Complete)

**Dokument stworzony:** 2026-04-30  
**Ostatnia zmiana:** 2026-05-21  
**Status:** Production-ready - Gotowy do implementacji
