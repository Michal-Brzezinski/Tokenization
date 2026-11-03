import textdistance

# ==============================================================================
# CZĘŚĆ F: WIĘCEJ MIAR PODOBIEŃSTWA (textdistance)
# ==============================================================================

print("=" * 75)
print("CZĘŚĆ F: PORÓWNANIE CZTERECH MIAR PODOBIEŃSTWA TEKSTU")
print("=" * 75)

# --------------------------------------------------
# 1. Wybór 3 dodatkowych miar i opis zasady działania:
#    1. Levenshtein (bazowa miara edycyjna)
#    2. Damerau-Levenshtein (edycyjna z transpozycją)
#    3. Jaro-Winkler (heurystyczna miara prefiksowa dla literówek)
#    4. Jaccard (miara zbiorowa, oparta na unikalnych znakach)
# --------------------------------------------------

descriptions = """
OPIS WYBRANYCH MIAR PODOBIEŃSTWA:

1. Odległość Levenshteina:
   - Klasyczna miara edycyjna.
   - Minimalna liczba operacji wstawienia (insertion), usunięcia (deletion)
     lub zamiany (substitution) pojedynczego znaku potrzebnych do przekształcenia
     jednego napisu w drugi. Każda operacja ma koszt 1.

2. Odległość Damerau-Levenshteina:
   - Rozszerzenie odległości Levenshteina o czwartą dopuszczalną operację:
     transpozycję (zamianę miejscami dwóch SĄSIEDNICH znaków, np. 'ab' -> 'ba').
   - W klasycznym Levenshteinie transpozycja kosztuje 2 (dwie substytucje),
     natomiast w Damerau-Levenshtein kosztuje tylko 1 operację.
   - Doskonale modeluje typowe błędy ludzkie (czeskie błędy / literówki przy szybkim pisaniu).

3. Podobieństwo Jaro-Winklera (Jaro-Winkler Similarity):
   - Miara z przedziału [0, 1], gdzie 1 oznacza identyczność, a 0 brak podobieństwa.
   - Bazuje na mierze Jaro, uwzględniającej liczbę wspólnych znaków w oknie dopasowania
     oraz liczbę transpozycji.
   - Dodatek Winklera premiuje napisy posiadające wspólny prefiks (początkowe do 4 znaków).
     Zaprojektowana specjalnie do porównywania nazwisk i danych osobowych w bazach danych.

4. Podobieństwo Jaccarda (Jaccard Similarity):
   - Miara zbiorowa (oparta na teorii zbiorów).
   - Oblicza iloraz mocy części wspólnej (przecięcia) do sumy zbiorów elementów:
     J(A, B) = |A iloczyn B| / |A suma B|.
   - W przypadku napisów traktuje tekst jako zbiór unikalnych znaków (lub n-gramów znakowych).
   - CAŁKOWICIE IGNORUJE kolejność znaków oraz ich wielokrotne powtórzenia.
"""

print(descriptions)

# --------------------------------------------------
# 2. Porównanie wyników 4 miar dla wymaganej pary oraz dobranych par testowych
# --------------------------------------------------

test_pairs = [
    {
        "s1": "Ala ma kota",
        "s2": "Ala posiada kota",
        "desc": "Para bazowa (wymiana słowa 'ma' na 'posiada')"
    },
    {
        "s1": "kot",
        "s2": "kto",
        "desc": "Transpozycja sąsiednich liter (czeski błąd)"
    },
    {
        "s1": "Warszawa",
        "s2": "Warszwa",
        "desc": "Literówka (usunięcie litery 'a' w środku słowa)"
    },
    {
        "s1": "żółć",
        "s2": "zolc",
        "desc": "Różnica wyłącznie w polskich znakach diakrytycznych"
    },
    {
        "s1": "pies",
        "s2": "seip",
        "desc": "Odwrócenie kolejności liter (te same litery, odwrotny szyk)"
    }
]

print("=" * 88)
print(f"| {'OPIS PARY':<22} | {'S1 / S2':<20} | {'LEV (d/s)':<10} | {'DAM (d/s)':<10} | {'JARO-WINKLER':<12} | {'JACCARD':<8} |")
print("=" * 88)

for p in test_pairs:
    s1, s2, desc = p["s1"], p["s2"], p["desc"]

    # Levenshtein
    lev_d = textdistance.levenshtein.distance(s1, s2)
    lev_s = textdistance.levenshtein.normalized_similarity(s1, s2)

    # Damerau-Levenshtein
    dam_d = textdistance.damerau_levenshtein.distance(s1, s2)
    dam_s = textdistance.damerau_levenshtein.normalized_similarity(s1, s2)

    # Jaro-Winkler
    jw_s = textdistance.jaro_winkler.similarity(s1, s2)

    # Jaccard
    jac_s = textdistance.jaccard.similarity(s1, s2)

    pair_str = f"'{s1}' vs '{s2}'"
    print(f"| {desc:<22} | {pair_str:<20} | {lev_d:2d} ({lev_s:.2f}) | {dam_d:2d} ({dam_s:.2f}) | {jw_s:^12.3f} | {jac_s:^8.3f} |")
    print("-" * 88)

# --------------------------------------------------
# 3. Wskazanie pary o największych rozbieżnościach i wyjaśnienie (Punkt 4)
# --------------------------------------------------

print("\n" + "=" * 75)
print("ANALIZA PARY O NAJWIĘKSZEJ ROZBIEŻNOŚCI WYNIKÓW:")
print("=" * 75)

conclusion = """
Para o najbardziej skrajnej rozbieżności miar:
-> 'kot' vs 'kto' (oraz 'pies' vs 'seip')

Wyniki dla pary 'kot' vs 'kto':
- Podobieństwo Jaccarda:        1.000  (100% - pełne dopasowanie!)
- Podobieństwo Jaro-Winklera:   0.556  (55.6%)
- Damerau-Levenshtein:          odległość = 1 (podobieństwo = 0.67)
- Levenshtein:                  odległość = 2 (podobieństwo = 0.33)

WYJAŚNIENIE PRZYCZYNY ROZBIEŻNOŚCI:
1. Dlaczego Jaccard daje 1.0 (maksimum)?
   Miara Jaccarda jest miarą ZBIOROWĄ. Zbiór unikalnych znaków napisu 'kot' to {'k', 'o', 't'}.
   Zbiór unikalnych znaków napisu 'kto' to również {'k', 'o', 't'}.
   Iloczyn zbiorów jest równy ich sumie, dlatego Jaccard uznaje te dwa słowa za absolutnie
   identyczne. Miara ta jest całkowicie 'ślepa' na kolejność i układ przestrzenny liter.

2. Dlaczego Levenshtein daje odległość aż 2 (podobieństwo tylko 0.33)?
   Klasyczny algorytm Levenshteina jest ściśle SEKWENCYJNY. Nie posiada operacji
   zamiany sąsiednich znaków, dlatego aby przekształcić 'kot' w 'kto', musi wykonać
   dwie niezależne operacje zamiany: 'o' -> 't' oraz 't' -> 'o'.

3. Rola Damerau-Levenshteina:
   Damerau-Levenshtein idealnie diagnozuje tę sytuację jako pojedynczą transpozycję
   sąsiadów ('ot' <-> 'to'), dając realistyczny koszt = 1 operacja.

Wniosek praktyczny:
Wybór miary podobieństwa musi zależeć od problemu. Do wykrywania literówek sekwencyjnych
najlepszy jest Damerau-Levenshtein lub Jaro-Winkler. Miara Jaccarda sprawdza się przy
porównywaniu zbiorów słów (np. dokumentów), ale jest zawodna na poziomie pojedynczych wyrazów.
"""

print(conclusion)
