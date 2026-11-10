import re
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from stop_words import get_stop_words

# ==============================================================================
# CZĘŚĆ G: STOP WORDS I NAZWY WŁASNE
# ==============================================================================

print("=" * 70)
print("CZĘŚĆ G: STOP WORDS ORAZ WYKRYWANIE NAZW WŁASNYCH (Pan Tadeusz)")
print("=" * 70)

# --------------------------------------------------
# Wczytanie tekstu
# --------------------------------------------------

with open("texts/pan-tadeusz-adam-mickiewicz.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Odcięcie stopki licencyjnej Wolnych Lektur (tak samo jak w części A)
raw_text = raw_text.split("\n-----\n")[0]

# --------------------------------------------------
# 1. Usunięcie stop words z tekstu polskiego
# --------------------------------------------------

# Pobranie listy polskich słów funkcyjnych
polish_stop_words = set(get_stop_words("polish"))
print(f"Liczba stop words w bibliotece stop-words (PL): {len(polish_stop_words)}")

# Oczyszczenie tekstu do postaci słów
cleaned_text = raw_text.lower()
cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

words_all = word_tokenize(cleaned_text)
words_filtered = [w for w in words_all if w not in polish_stop_words and len(w) > 1]

print(f"Liczba wyrazów przed filtracją (Część A): {len(words_all)}")
print(f"Liczba wyrazów po usunięciu stop words:   {len(words_filtered)}")
print(f"Usunięto słów funkcyjnych:               {len(words_all) - len(words_filtered)} "
      f"({(len(words_all) - len(words_filtered)) / len(words_all) * 100:.2f}%)")

# --------------------------------------------------
# 2. Powtórzenie analizy n-gramów na tekście bez stop words
# --------------------------------------------------

unigrams = list(ngrams(words_filtered, 1))
bigrams = list(ngrams(words_filtered, 2))
trigrams = list(ngrams(words_filtered, 3))

unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

top_unigrams = unigram_freq.most_common(10)
top_bigrams = bigram_freq.most_common(10)
top_trigrams = trigram_freq.most_common(10)

print("\n10 NAJCZĘSTSZYCH UNIGRAMÓW (BEZ STOP WORDS):")
for u, c in top_unigrams:
    print(f"  {u[0]:<15} {c}")

print("\n10 NAJCZĘSTSZYCH BIGRAMÓW (BEZ STOP WORDS):")
for b, c in top_bigrams:
    print(f"  {' '.join(b):<25} {c}")

print("\n10 NAJCZĘSTSZYCH TRIGRAMÓW (BEZ STOP WORDS):")
for t, c in top_trigrams:
    print(f"  {' '.join(t):<35} {c}")

# Rysowanie histogramów
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Unigramy
labels_ug = [u[0] for u, _ in top_unigrams]
values_ug = [count for _, count in top_unigrams]
axes[0].bar(labels_ug, values_ug, color="royalblue")
axes[0].set_title("10 najczęstszych unigramów (bez stop words)")
axes[0].tick_params(axis="x", rotation=45)

# Bigramy
labels_bg = [" ".join(b) for b, _ in top_bigrams]
values_bg = [count for _, count in top_bigrams]
axes[1].bar(labels_bg, values_bg, color="darkorange")
axes[1].set_title("10 najczęstszych bigramów (bez stop words)")
axes[1].tick_params(axis="x", rotation=45)

# Trigramy
labels_tg = [" ".join(t) for t, _ in top_trigrams]
values_tg = [count for _, count in top_trigrams]
axes[2].bar(labels_tg, values_tg, color="mediumseagreen")
axes[2].set_title("10 najczęstszych trigramów (bez stop words)")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("wykres_czesc_g.png", dpi=300)
plt.close()
print("\n[OK] Zapisano wykres n-gramów bez stop words do pliku 'wykres_czesc_g.png'.")

# --------------------------------------------------
# Ilościowe porównanie obu wariantów (Punkt 3)
# --------------------------------------------------

def ngram_stats(words, n):
    freq = Counter(ngrams(words, n))
    total = max(len(words) - n + 1, 1)
    top10 = sum(c for _, c in freq.most_common(10))
    repeated = sum(1 for c in freq.values() if c > 1)
    return len(freq), top10 / total * 100, repeated

print("\nPORÓWNANIE WARIANTÓW (ze stop words vs bez stop words):")
print(f"| {'n':<2} | {'unikalne ze SW':>14} | {'unikalne bez SW':>15} | {'top-10 ze SW':>12} | {'top-10 bez SW':>13} | {'powt. ze SW':>11} | {'powt. bez SW':>12} |")
for n in (1, 2, 3):
    u1, t1, r1 = ngram_stats(words_all, n)
    u2, t2, r2 = ngram_stats(words_filtered, n)
    print(f"| {n:<2} | {u1:>14} | {u2:>15} | {t1:>11.2f}% | {t2:>12.2f}% | {r1:>11} | {r2:>12} |")

# --------------------------------------------------
# 3. Odnalezienie nazw własnych wyrażeniem regularnym (Punkty 4 i 5)
# --------------------------------------------------

print("\n" + "=" * 70)
print("WYKRYWANIE NAZW WŁASNYCH WYRAŻENIEM REGULARNYM")
print("=" * 70)

# Pan Tadeusz jest pisany wierszem – KAŻDY wers zaczyna się wielką literą, także
# w środku zdania. Samo pominięcie pierwszego słowa zdania nie wystarcza (do nazw
# trafiłyby np. „Potem”, „Tymczasem”, „Wtem”, „Rzekł” z początków wersów).
#
# Wyrażenie regularne:
#   (?<=[a-ząćęłńóśźż,;)] )      – lookbehind: tuż przed słowem jest JEDNA spacja,
#                                  a przed nią mała litera, przecinek, średnik lub „)”.
#                                  Wyklucza to: początek wersu (przed słowem jest \n
#                                  lub wcięcie), początek zdania (przed spacją jest
#                                  . ! ? …) oraz początek wypowiedzi (po „—” lub „:”).
#   [A-ZĄĆĘŁŃÓŚŹŻ]               – pierwsza litera wielka (z polskimi diakrytykami),
#   [a-ząćęłńóśźż]+               – reszta wyrazu małymi literami (odrzuca skróty
#                                  pisane wersalikami, np. „ISBN”).
NAME_RE = re.compile(r"(?<=[a-ząćęłńóśźż,;)] )[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+")

proper_names = NAME_RE.findall(raw_text)
names_counter = Counter(proper_names)
top_10_names = names_counter.most_common(10)

print(f"Dopasowań: {len(proper_names)}, unikalnych form: {len(names_counter)}\n")
h1, h2 = "NAZWA WŁASNA (forma)", "LICZBA WYSTĄPIEŃ"
print(f"|{h1:^25} | {h2:^18}|")
print("=" * 47)
for name, cnt in top_10_names:
    print(f"|{name:^25} | {cnt:^18}|")
    print("-" * 47)

# Porównanie z metodą naiwną: „pierwsze słowo zdania odpada, reszta z wielkiej litery”
naive = []
for s in sent_tokenize(raw_text):
    for w in re.findall(r"[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+", s)[1:]:
        if w.lower() not in polish_stop_words:
            naive.append(w)
naive_counter = Counter(naive)
false_pos = [w for w in naive_counter if w not in names_counter]
print(f"\nMetoda naiwna (tylko pominięcie 1. słowa zdania): {len(naive)} dopasowań, "
      f"{len(naive_counter)} unikalnych form.")
print("Przykładowe formy znalezione WYŁĄCZNIE metodą naiwną (początki wersów):")
print("  " + ", ".join(sorted(false_pos, key=lambda w: -naive_counter[w])[:15]))

# --------------------------------------------------
# Grupowanie form fleksyjnych (Tadeusz / Tadeusza / Tadeuszowi ...) lematem Morfeusza2
# --------------------------------------------------
try:
    import morfeusz2
    morf = morfeusz2.Morfeusz()

    def name_lemma(form):
        interps = [r[2] for r in morf.analyse(form)]
        # preferujemy interpretację, której lemat zaczyna się wielką literą (nazwa własna)
        for i in interps:
            lemma = i[1].split(":")[0]
            if lemma[:1].isupper():
                return lemma
        return interps[0][1].split(":")[0].capitalize()

    lemma_counter = Counter()
    for form, cnt in names_counter.items():
        lemma_counter[name_lemma(form)] += cnt

    print("\n10 NAJCZĘSTSZYCH NAZW WŁASNYCH PO ZGRUPOWANIU FORM (lemat Morfeusz2):")
    for lemma, cnt in lemma_counter.most_common(10):
        forms = sorted((f for f in names_counter if name_lemma(f) == lemma),
                       key=lambda f: -names_counter[f])
        print(f"  {lemma:<12} {cnt:>4}   ← {', '.join(f'{f} ({names_counter[f]})' for f in forms[:4])}")
except ImportError:
    print("\n(morfeusz2 niedostępny – pomijam grupowanie form)")

# ==============================================================================
# ODPOWIEDZI NA PYTANIA I WNIOSKI (PUNKTY 3, 4 I 5)
# ==============================================================================
"""
WNIOSKI I ODPOWIEDZI NA PYTANIA:

1. Porównanie wariantów (ze stop words vs bez stop words):
   - Różnice są bardzo istotne. Lista stop-words usuwa ok. 36% wszystkich wyrazów.
     W części A najczęstsze bigramy to wyłącznie słowa funkcyjne ('się w', 'i z', 'się z').
   - Po usunięciu stop words w n-gramach pojawiają się terminy fabularne:
     * Bohaterowie: 'rzekł sędzia', 'ksiądz robak', 'rzekł hrabia', 'kapitan ryków',
       'jenerał dąbrowski'.
     * Tytuły i zwroty: 'jaśnie wielmożny', 'pro publico bono'.
     * Motywy fabularne: 'kusego sokoła' (spór o charty), 'hejże soplicę' (zajazd).
   - Trigramy bez stop words są sztuczne ('wieku urzędu zaszczyt') – po usunięciu słów
     funkcyjnych łączą wyrazy, które w oryginale nie stały obok siebie.
   - STYL lepiej pokazuje wariant ZE STOP WORDS (rytm, składnia, spójniki, 'się').
     TREŚĆ lepiej pokazuje wariant BEZ STOP WORDS (bohaterowie, miejsca, wątki).

2. Wyrażenie regularne do nazw własnych – opis w komentarzu przy NAME_RE.
   Kluczowy jest lookbehind: słowo z wielkiej litery musi stać w ŚRODKU wersu, za
   małą literą lub przecinkiem. To odrzuca początki zdań i wersów kosztem pominięcia
   nazw, które akurat stoją na początku wersu (lepsza precyzja, gorsza kompletność).
   Pozostałe błędy to rzeczowniki pisane wielką literą z szacunku ('Bóg', 'Pan')
   i tytuły używane jak imiona ('Sędzia', 'Hrabia', 'Wojski') – w Panu Tadeuszu
   pełnią one jednak rolę imion bohaterów.
"""

