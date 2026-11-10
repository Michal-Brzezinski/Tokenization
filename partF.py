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
   - textdistance domyślnie (qval=1, as_set=False) traktuje napis jako MULTIZBIÓR znaków:
     liczy część wspólną i sumę z uwzględnieniem krotności (np. 'a' występujące 3 razy).
   - CAŁKOWICIE IGNORUJE kolejność znaków – liczy się tylko „worek liter”.
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

    # Rozbieżność miar = rozstęp (max - min) znormalizowanych podobieństw w [0, 1]
    sims = [lev_s, dam_s, jw_s, jac_s]
    p["spread"] = max(sims) - min(sims)

print("\nROZBIEŻNOŚĆ MIAR (max - min znormalizowanego podobieństwa):")
for p in sorted(test_pairs, key=lambda p: -p["spread"]):
    print(f"  '{p['s1']}' vs '{p['s2']}': {p['spread']:.3f}")

# --------------------------------------------------
# 3. Wskazanie pary o największych rozbieżnościach i wyjaśnienie (Punkt 4)
# --------------------------------------------------

print("\n" + "=" * 75)
print("ANALIZA PARY O NAJWIĘKSZEJ ROZBIEŻNOŚCI WYNIKÓW:")
print("=" * 75)

conclusion = """
Para o największej rozbieżności miar (rozstęp = 1.000):
-> 'pies' vs 'seip'   (druga: 'kot' vs 'kto', rozstęp = 0.667)

Wyniki dla pary 'pies' vs 'seip':
- Jaccard:             1.000  – te same litery, każda raz → identyczny „worek liter”
- Jaro-Winkler:        0.500  – wszystkie 4 litery są w oknie dopasowania, ale 2 transpozycje
                                i brak wspólnego prefiksu (brak premii Winklera)
- Damerau-Levenshtein: odl. 3 (podob. 0.25) – jedna transpozycja ('ie' -> 'ei') + 2 zamiany
- Levenshtein:         odl. 4 (podob. 0.00) – każda pozycja inna, 4 zamiany

WYJAŚNIENIE PRZYCZYNY ROZBIEŻNOŚCI:
1. Jaccard jest miarą ZBIOROWĄ (multizbiór znaków). Multizbiory {p, i, e, s} dla obu
   napisów są identyczne, więc część wspólna = suma i wynik = 1. Miara jest „ślepa”
   na kolejność liter – dla odwróconego słowa daje maksimum.
2. Levenshtein jest ściśle POZYCYJNY i nie ma operacji transpozycji: żadna litera
   nie stoi na swoim miejscu, więc potrzeba 4 zamian – podobieństwo spada do zera.
3. Damerau-Levenshtein dodaje transpozycję sąsiadów o koszcie 1, dzięki czemu
   oszczędza jedną operację (3 zamiast 4).
4. Jaro-Winkler mieści się pośrodku: liczy wspólne znaki w oknie, karze za transpozycje,
   a premia za prefiks nie działa, bo pierwsze litery się różnią.
Te same mechanizmy widać w parze 'kot' vs 'kto': Jaccard = 1.0, Levenshtein = 2 zamiany
(0.33), Damerau-Levenshtein = 1 transpozycja (0.67).

Ciekawy jest też przypadek 'żółć' vs 'zolc': wszystkie miary dają 0, bo dla komputera
'ż' i 'z' to zupełnie różne znaki Unicode. Wykrywanie takich różnic wymaga normalizacji
(np. usunięcia diakrytyków) PRZED liczeniem odległości.

Wniosek praktyczny:
Wybór miary podobieństwa musi zależeć od problemu. Do wykrywania literówek sekwencyjnych
najlepszy jest Damerau-Levenshtein lub Jaro-Winkler. Miara Jaccarda sprawdza się przy
porównywaniu zbiorów słów (np. dokumentów), ale jest zawodna na poziomie pojedynczych wyrazów.
"""

print(conclusion)
