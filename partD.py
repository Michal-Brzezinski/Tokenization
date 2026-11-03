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

# Słowa dobrane bezpośrednio z korpusu Hamleta (z potwierdzonym występowaniem w tekście):
selected_words = [
    "father",       # występuje 59 razy
    "heaven",       # występuje 57 razy
    "nature",       # występuje 23 razy
    "matter",       # występuje 21 razy
    "general",      # występuje 14 razy
    "history",      # występuje 7 razy
    "philosophy",   # występuje 3 razy
    "friendship"    # występuje 2 razy
]

h1, h2, h3 = "SŁOWO (Z HAMLETA)", "PORTER STEMMER", "LANCASTER STEMMER"
print(f"\nPORÓWNANIE WYNIKÓW DLA SŁÓW ZIDENTYFIKOWANYCH W TEKŚCIE HAMLETA:")
print("=" * 65)
print(f"|{h1:^21} | {h2:^19} | {h3:^19}|")
print("=" * 65)

for word in selected_words:
    p_stem = porter.stem(word)
    l_stem = lancaster.stem(word)
    print(f"|{word:^21} | {p_stem:^19} | {l_stem:^19}|")
    print("-" * 65)

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

over_examples = [
    ("university", "universe"),
    ("organization", "organ"),
    ("policy", "police")
]

for w1, w2 in over_examples:
    print(f"Para słów: '{w1}' i '{w2}'")
    print(f"  - Porter:    '{w1}' -> {porter.stem(w1):<10} | '{w2}' -> {porter.stem(w2)}")
    print(f"  - Lancaster: '{w1}' -> {lancaster.stem(w1):<10} | '{w2}' -> {lancaster.stem(w2)}")
    print()

print("[B] UNDERSTEMMING (niedostateczne obcinanie):")
print("Dwie formy tego samego pojęcia/wyrazu NIE schodzą się do jednego rdzenia.\n")

under_examples = [
    ("europe", "european"),
    ("matrix", "matrices"),
    ("create", "creative")
]

for w1, w2 in under_examples:
    print(f"Para słów: '{w1}' i '{w2}'")
    print(f"  - Porter:    '{w1}' -> {porter.stem(w1):<10} | '{w2}' -> {porter.stem(w2)}")
    print(f"  - Lancaster: '{w1}' -> {lancaster.stem(w1):<10} | '{w2}' -> {lancaster.stem(w2)}")
    print()
