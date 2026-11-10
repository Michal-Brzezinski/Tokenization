import nltk
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# ==============================================================================
# CZĘŚĆ D: PORÓWNANIE STEMMERÓW DLA JĘZYKA ANGIELSKIEGO
# ==============================================================================

# --------------------------------------------------
# 1. Wczytanie tekstu dramatu Hamlet
# --------------------------------------------------

with open("texts/hamlet-william-shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizacja i ekstrakcja unikalnych słów
tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
unique_tokens = sorted(list(set(tokens)))

# Inicjalizacja dwóch różnych stemmerów
porter = PorterStemmer()       # algorytm kaskadowy, konserwatywny
lancaster = LancasterStemmer() # algorytm iteracyjny, bardzo agresywny

print("=" * 70)
print("CZĘŚĆ D: PORÓWNANIE STEMMERÓW (PORTER vs LANCASTER)")
print("=" * 70)

# --------------------------------------------------
# 2. Znalezienie słów, dla których algorytmy dają różne rdzenie
# --------------------------------------------------

diff_words = []
for word in unique_tokens:
    p_stem = porter.stem(word)
    l_stem = lancaster.stem(word)
    if p_stem != l_stem:
        diff_words.append((word, p_stem, l_stem))

from collections import Counter
token_counts = Counter(tokens)

def in_hamlet(word):
    return f"{word} [{token_counts.get(word, 0)}x w Hamlecie]"

# Słowa dobrane bezpośrednio z korpusu Hamleta (liczba wystąpień liczona z tekstu):
selected_words = [
    "father", "heaven", "nature", "matter",
    "general", "history", "philosophy", "friendship"
]

h1, h2, h3 = "SŁOWO (Z HAMLETA)", "PORTER STEMMER", "LANCASTER STEMMER"
print(f"\nPORÓWNANIE WYNIKÓW DLA SŁÓW ZIDENTYFIKOWANYCH W TEKŚCIE HAMLETA:")
print("=" * 72)
print(f"|{h1:^28} | {h2:^19} | {h3:^19}|")
print("=" * 72)

for word in selected_words:
    p_stem = porter.stem(word)
    l_stem = lancaster.stem(word)
    print(f"|{in_hamlet(word):^28} | {p_stem:^19} | {l_stem:^19}|")
    print("-" * 72)

print(f"\nŁączna liczba unikalnych słów w tekście Hamleta z różnymi rdzeniami: {len(diff_words)}")

# --------------------------------------------------
# 3. Przykłady nadmiernego obcinania (Overstemming)
#    i niedostatecznego obcinania (Understemming)
# --------------------------------------------------

print("\n" + "=" * 70)
print("PRZYKŁADY OVERSTEMMINGU I UNDERSTEMMINGU")
print("=" * 70)

print("\n[A] OVERSTEMMING (nadmierne obcinanie):")
print("Dwa semantycznie różne słowa zostają zredukowane do tego samego rdzenia.\n")

# Pary znalezione w tekście Hamleta (+ jedna klasyczna para spoza tekstu)
over_examples = [
    ("general", "generous"),   # Porter: gener / gener   – „ogólny” vs „hojny”
    ("ear", "earth"),          # Lancaster: ear / ear    – „ucho” vs „ziemia”
    ("not", "notice"),         # Lancaster: not / not    – przeczenie vs „zauważyć”
    ("mad", "made"),           # Lancaster: mad / mad    – „szalony” vs „zrobiony”
    ("university", "universe") # klasyczny przykład („universe” nie występuje w tekście)
]

for w1, w2 in over_examples:
    print(f"Para słów: '{in_hamlet(w1)}' i '{in_hamlet(w2)}'")
    print(f"  - Porter:    '{w1}' -> {porter.stem(w1):<10} | '{w2}' -> {porter.stem(w2)}")
    print(f"  - Lancaster: '{w1}' -> {lancaster.stem(w1):<10} | '{w2}' -> {lancaster.stem(w2)}")
    print()

print("[B] UNDERSTEMMING (niedostateczne obcinanie):")
print("Dwie formy tego samego pojęcia/wyrazu NIE schodzą się do jednego rdzenia.\n")

under_examples = [
    ("die", "died"),         # Lancaster: die / died   (Porter scala poprawnie)
    ("think", "thought"),    # forma nieregularna – żaden stemmer nie scala
    ("tell", "told"),        # forma nieregularna – żaden stemmer nie scala
    ("speak", "spoke"),      # forma nieregularna – żaden stemmer nie scala
    ("matrix", "matrices")   # klasyczny przykład spoza tekstu
]

for w1, w2 in under_examples:
    print(f"Para słów: '{in_hamlet(w1)}' i '{in_hamlet(w2)}'")
    print(f"  - Porter:    '{w1}' -> {porter.stem(w1):<10} | '{w2}' -> {porter.stem(w2)}")
    print(f"  - Lancaster: '{w1}' -> {lancaster.stem(w1):<10} | '{w2}' -> {lancaster.stem(w2)}")
    print()

# ==============================================================================
# WNIOSKI
# ==============================================================================
"""
1. Który wynik jest sensowniejszy?
   Porter jest konserwatywny (father -> father, matter -> matter, friendship -> friendship),
   Lancaster bardzo agresywny (matter -> mat, general -> gen, nature -> nat).
   Rdzenie Portera są sensowniejsze: nadal da się z nich odczytać wyraz, a różne leksemy
   rzadziej się zlewają. Rdzeń 'mat' Lancastera mógłby pochodzić od matter, mat, match
   czy material, więc w wyszukiwarce dałby dużo fałszywych trafień. Lancaster ma przewagę
   tylko tam, gdzie chcemy maksymalnie zmniejszyć słownik kosztem precyzji.

2. Overstemming (nadmierne obcinanie) – przykłady z Hamleta:
   - general / generous -> 'gener' (Porter) – „ogólny” i „hojny” dostają ten sam rdzeń,
   - ear / earth -> 'ear', not / notice -> 'not', mad / made -> 'mad' (Lancaster).

3. Understemming (niedostateczne obcinanie) – przykłady z Hamleta:
   - die / died -> 'die' / 'died' (Lancaster; Porter scala poprawnie do 'die'),
   - think / thought, tell / told, speak / spoke – formy nieregularne (ablaut), których
     żaden stemmer oparty na obcinaniu sufiksów nie połączy; potrzebna jest lematyzacja.
"""
