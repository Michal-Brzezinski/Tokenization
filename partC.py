import textdistance

# --------------------------------------------------
# Dane wejściowe (Punkt 1)
# --------------------------------------------------

s1 = "Ala ma kota"
s2 = "Ala posiada kota"

print("=" * 60)
print("CZĘŚĆ C: BADANIE ODLEGŁOŚCI EDYCYJNEJ LEVENSHTEINA")
print("=" * 60)
print(f"Zdanie 1: \"{s1}\" (długość: {len(s1)} znaków)")
print(f"Zdanie 2: \"{s2}\" (długość: {len(s2)} znaków)")

# --------------------------------------------------
# Obliczenie odległości Levenshteina (Punkt 2)
# --------------------------------------------------

distance = textdistance.levenshtein.distance(s1, s2)
similarity = textdistance.levenshtein.similarity(s1, s2)
normalized_dist = textdistance.levenshtein.normalized_distance(s1, s2)
normalized_sim = textdistance.levenshtein.normalized_similarity(s1, s2)

print("\nWYNIKI:")
print(f"- Odległość Levenshteina: {distance}")
print(f"- Podobieństwo bezwzględne: {similarity}")
print(f"- Znormalizowana odległość: {normalized_dist:.4f}")
print(f"- Znormalizowane podobieństwo: {normalized_sim:.4f}")

# --------------------------------------------------
# Wyjaśnienie wyniku i operacji edycyjnych (Punkt 3)
# --------------------------------------------------

print("\n" + "=" * 60)
print("SZCZEGÓŁOWE ROZPISANIE OPERACJI EDYCYJNYCH:")
print("=" * 60)

explanation = """
Odległość Levenshteina między zdaniami wynosi dokładnie 6.
Oznacza to, że minimalna liczba elementarnych operacji edycyjnych
(wstawienie, usunięcie, zamiana znaku) potrzebnych do przekształcenia
zdania pierwszego w drugie wynosi 6.

1. Analiza wspólnych fragmentów:
   - Wspólny prefiks: "Ala " (4 znaki) -> koszt: 0 operacji
   - Wspólny sufiks: " kota" (5 znaków) -> koszt: 0 operacji
   - Różnica występuje wyłącznie między słowami: "ma" -> "posiada".

2. Przekształcenie słowa 'ma' (2 znaki) w 'posiada' (7 znaków):
   Krok 1: Zamiana (substytucja) 'm' -> 'p'       [koszt = 1]
   Krok 2: Wstawienie (insercja)   'o'            [koszt = 1]
   Krok 3: Wstawienie (insercja)   's'            [koszt = 1]
   Krok 4: Wstawienie (insercja)   'i'            [koszt = 1]
   Krok 5: Zachowanie litery       'a' (match)    [koszt = 0]
   Krok 6: Wstawienie (insercja)   'd'            [koszt = 1]
   Krok 7: Wstawienie (insercja)   'a'            [koszt = 1]

Suma operacji:
- 1 zamiana (substytucja: 'm' -> 'p')
- 5 wstawień (insercji: 'o', 's', 'i', 'd', 'a')
- 0 usunięć (delecji)
-----------------------------------------------
ŁĄCZNIE: 6 operacji edycyjnych o łącznym koszcie = 6.

Uwaga:
Gdyby algorytm nie wykorzystał wspólnej litery 'a', musiałby usunąć 'ma' (2 operacje)
i wstawić 'posiada' (7 operacji), co dałoby łącznie 9 operacji. Dzięki dynamicznemu
programowaniu i dopasowaniu litery 'a' algorytm osiąga optymalną ścieżkę równą 6.
"""

print(explanation)
