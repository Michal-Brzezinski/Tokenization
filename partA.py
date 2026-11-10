import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

import matplotlib.pyplot as plt

# --------------------------------------------------
# Wczytanie pliku
# --------------------------------------------------

with open("texts/pan-tadeusz-adam-mickiewicz.txt", "r", encoding="utf-8") as f:
# with open("texts/rozmyslania-marek-aureliusz.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Odcięcie stopki licencyjnej Wolnych Lektur (zaczyna się od linii "-----"),
# żeby metadane nie zaburzały statystyk n-gramów
raw_text = raw_text.split("\n-----\n")[0]

# --------------------------------------------------
# Tokenizacja na zdania (wykonana na surowym tekście,
# aby zachować znaki interpunkcyjne kończące zdania: ., !, ?)
# --------------------------------------------------

sentences = sent_tokenize(raw_text)
print(f"Liczba zdań: {len(sentences)}")

# --------------------------------------------------
# Czyszczenie tekstu na potrzeby tokenizacji wyrazów i n-gramów
# --------------------------------------------------

# 1. zamiana liter na małe
text = raw_text.lower()

# 2. usunięcie znaków specjalnych i interpunkcji
text = re.sub(r"[^\w\s]", " ", text)

# 3. zwinięcie wielokrotnych białych znaków
text = re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------
# Tokenizacja na wyrazy
# --------------------------------------------------

words = word_tokenize(text)
print(f"Liczba wyrazów: {len(words)}")

# --------------------------------------------------
# Generowanie n-gramów słownych
# --------------------------------------------------

unigrams = list(ngrams(words, 1))
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))

print(f"Unigramy: {len(unigrams)}")
print(f"Bigramy: {len(bigrams)}")
print(f"Trigramy: {len(trigrams)}")

# --------------------------------------------------
# Liczenie częstości występowania (Counter)
# --------------------------------------------------

unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

print("\n10 najczęstszych unigramów:")
top_unigrams = unigram_freq.most_common(10)
for unigram, count in top_unigrams:
    print(unigram, count)

print("\n10 najczęstszych bigramów:")
top_bigrams = bigram_freq.most_common(10)
for bigram, count in top_bigrams:
    print(bigram, count)

print("\n10 najczęstszych trigramów:")
top_trigrams = trigram_freq.most_common(10)
for trigram, count in top_trigrams:
    print(trigram, count)

# --------------------------------------------------
# Histogramy n-gramów (matplotlib)
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Unigramy
labels_ug = [unigram[0] for unigram, _ in top_unigrams]
values_ug = [count for _, count in top_unigrams]
axes[0].bar(labels_ug, values_ug, color="steelblue")
axes[0].set_title("10 najczęstszych unigramów")
axes[0].tick_params(axis="x", rotation=45)

# Bigramy
labels_bg = [" ".join(bigram) for bigram, _ in top_bigrams]
values_bg = [count for _, count in top_bigrams]
axes[1].bar(labels_bg, values_bg, color="coral")
axes[1].set_title("10 najczęstszych bigramów")
axes[1].tick_params(axis="x", rotation=45)

# Trigramy
labels_tg = [" ".join(trigram) for trigram, _ in top_trigrams]
values_tg = [count for _, count in top_trigrams]
axes[2].bar(labels_tg, values_tg, color="seagreen")
axes[2].set_title("10 najczęstszych trigramów")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("wykres_czesc_a.png", dpi=300)
plt.close()

# --------------------------------------------------
# Porównanie rozkładów unigramów, bigramów i trigramów (do pytania 3)
# --------------------------------------------------

print("\nSTATYSTYKI ROZKŁADÓW N-GRAMÓW:")
print(f"| {'n-gram':<9} | {'wystąpienia':>11} | {'unikalne':>9} | {'hapax (1x)':>10} | {'% hapax':>7} | {'top-10 / całość':>15} |")
for name, freq, total in [("unigramy", unigram_freq, len(unigrams)),
                          ("bigramy", bigram_freq, len(bigrams)),
                          ("trigramy", trigram_freq, len(trigrams))]:
    hapax = sum(1 for c in freq.values() if c == 1)
    top10 = sum(c for _, c in freq.most_common(10))
    print(f"| {name:<9} | {total:>11} | {len(freq):>9} | {hapax:>10} | {hapax / len(freq) * 100:>6.1f}% | {top10 / total * 100:>14.2f}% |")

# Wykres rangowo-częstościowy (prawo Zipfa) w skali log-log
fig, ax = plt.subplots(figsize=(8, 5))
for name, freq, color in [("unigramy", unigram_freq, "steelblue"),
                          ("bigramy", bigram_freq, "coral"),
                          ("trigramy", trigram_freq, "seagreen")]:
    counts = sorted(freq.values(), reverse=True)
    ax.loglog(range(1, len(counts) + 1), counts, label=name, color=color, linewidth=1.8)
ax.set_xlabel("Ranga n-gramu (skala log)")
ax.set_ylabel("Liczba wystąpień (skala log)")
ax.set_title("Rozkład rangowo-częstościowy n-gramów (Pan Tadeusz)")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig("wykres_czesc_a_zipf.png", dpi=200)
plt.close()

# ==============================================================================
# ODPOWIEDZI NA PYTANIA (PUNKT 7)
# ==============================================================================
"""
1. Jakie bigramy dominują w analizowanym tekście?
   Odpowiedź:
   W analizowanym tekście dominują połączenia wyrazów funkcyjnych (tzw. stop words):
   spójników, przyimków oraz zaimków, a w szczególności zaimka zwrotnego „się”
   (np. „się w” [76], „i z” [72], „się z” [68], „się i” [61], „i w” [55], „się na” [47]).
   Bigramy te pełnią rolę łączników gramatycznych i konstrukcyjnych w zdaniach,
   nie niosąc same w sobie samodzielnego znaczenia semantycznego (treściowego).

2. Czy trigramy pokazują powtarzalne zwroty charakterystyczne dla autora?
   Odpowiedź:
   Tak, w przeciwieństwie do bigramów, w trigramach wyraźnie ujawniają się zwroty
   i motywy charakterystyczne dla fabuły oraz stylu Adama Mickiewicza:
   - „hejże na soplicę” (6 wystąpień) – zawołanie bitewne szlachty podczas zajazdu,
   - „ostatniego z horeszków” (5 wystąpień) – powracający motyw rodowy Hrabiego i zamku,
   - „za mych czasów” (5 wystąpień) – zwrot nostalgiczny, typowy dla gawędziarskiego tonu starszyzny,
   - „pro publico bono” (5 wystąpień) – makaronizm łaciński używany przez bohaterów,
   - „w domu sędziego” (4 wystąpienia) – wskazanie głównego miejsca wydarzeń,
   - „idąc kłaniał się” (5 wystąpień) – odzwierciedlenie szlacheckiej etykiety i obyczajowości.
   Trigramy znacznie lepiej niż unigramy i bigramy oddają specyfikę i koloryt językowy dzieła.

3. Jak różni się rozkład unigramów od rozkładu bigramów?
   Odpowiedź:
   - Częstość bezwzględna: Najczęstszy unigram („i”) występuje aż 2128 razy,
     podczas gdy najczęstszy bigram („się w”) jedynie 76 razy. Oznacza to gwałtowny
     spadek maksymalnej częstości (o niemal 96%).
   - Kształt rozkładu: Rozkład unigramów jest silnie skośny i wykazuje cechy klasycznego
     prawa Zipfa – 10 najczęstszych słów pokrywa 16,4% całego tekstu.
     Rozkład bigramów jest znacznie bardziej spłaszczony i rozproszony (ang. long tail):
     10 najczęstszych bigramów to tylko 0,76% wszystkich wystąpień, a 92,7% unikalnych
     bigramów pojawia się dokładnie raz (dla unigramów: 63,1%, dla trigramów: 99,1%).
     Przestrzeń potencjalnych par wyrazów rośnie kwadratowo (V^2), więc przy stałej
     długości tekstu prawie każda para jest „nowa”. Na wykresie log-log
     (wykres_czesc_a_zipf.png) krzywe bigramów i trigramów leżą dużo niżej i szybciej
     przechodzą w długie „plateau” n-gramów występujących jeden raz.
"""
