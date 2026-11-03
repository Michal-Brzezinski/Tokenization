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
# 3. Odnalezienie nazw własnych wyrażeniem regularnym (Punkty 4 i 5)
# --------------------------------------------------

print("\n" + "=" * 70)
print("WYKRYWANIE NAZW WŁASNYCH WYRAŻENIEM REGULARNYM")
print("=" * 70)

# Krok 1: Podział na zdania, aby wyeliminować pierwsze słowo każdego zdania
sentences = sent_tokenize(raw_text)
proper_names = []

# Wyrażenie regularne dopasowujące słowa polskie (z polskimi znakami diakrytycznymi)
word_pattern = re.compile(r"[A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż]+")

for s in sentences:
    words_in_sentence = word_pattern.findall(s)
    if len(words_in_sentence) > 1:
        # Sprawdzamy słowa od drugiego (indeks 1 wzwyż) - wewnątrz zdania
        for word in words_in_sentence[1:]:
            # Słowo zaczyna się wielką literą, a reszta to małe litery
            if word[0].isupper() and word[1:].islower():
                # Odrzucamy słowa funkcyjne z początku wersów (np. Lecz, Gdy, Ale, Na),
                # które znalazły się wewnątrz zdania z powodu zapisu wierszem
                if word.lower() not in polish_stop_words:
                    proper_names.append(word)

names_counter = Counter(proper_names)
top_10_names = names_counter.most_common(10)

h1, h2 = "NAZWA WŁASNA / POSTAĆ", "LICZBA WYSTĄPIEŃ"
print(f"|{h1:^25} | {h2:^18}|")
print("=" * 47)
for name, cnt in top_10_names:
    print(f"|{name:^25} | {cnt:^18}|")
    print("-" * 47)

# ==============================================================================
# ODPOWIEDZI NA PYTANIA I WNIOSKI (PUNKTY 3, 4 I 5)
# ==============================================================================
"""
WNIOSKI I ODPOWIEDZI NA PYTANIA:

1. Porównanie wariantów (ze stop words vs bez stop words):
   - Różnice są olbrzymie i fundamentalne. W części A najczęstszymi bigramami były
     wyłącznie słowa funkcyjne ('się w', 'i z', 'się z', 'się i').
   - Po usunięciu stop words w n-gramach pojawiają się kluczowe terminy fabularne:
     * Bohaterowie: 'rzekł sędzia', 'ksiądz robak', 'rzekł hrabia', 'kapitan ryków',
       'jenerał dąbrowski'.
     * Związki frazeologiczne i tytuły: 'jaśnie wielmożny', 'pro publico bono'.
     * Motywy fabularne: 'kusego sokoła' (spór o charty), 'hejże soplicę' (zajazd).
   - Który wariant lepiej pokazuje TREŚĆ, a który STYL?
     * STYL tekstu: Zdecydowanie lepiej pokazuje wariant ZE STOP WORDS. To słowa
       funkcyjne, zaimki i spójniki determinują rytm, dynamikę trzynastozgłoskowca,
       gawędziarski charakter epopei i archaizowaną składnię Mickiewicza.
     * TREŚĆ (semantykę) tekstu: Zdecydowanie lepiej pokazuje wariant BEZ STOP WORDS.
       Usuwa szum syntaktyczny i eksponuje to, o czym naprawdę jest utwór – głównych
       bohaterów, ich dialogi, tytuły i osie konfliktu.

2. Omówienie wyrażenia regularnego i metody wykrywania nazw własnych:
   - Zastosowano podejście dwuetapowe:
     1. Podział na zdania (sent_tokenize) – odrzucenie pierwszego słowa zdania
        (pozycja [0]), w której wielka litera wynika wyłącznie z reguł interpunkcyjnych.
     2. Regex: r'[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+' weryfikujący kapitalizację pierwszej litery
        i małe litery w reszcie wyrazu, z filtracją słów funkcyjnych (np. 'Gdy', 'Lecz'),
        które w tekście poetyckim rozpoczynają nowy wers mimo trwania zdania.
   - Wyniki: Zidentyfikowano czołowych bohaterów Pana Tadeusza:
     Sędzia (98), Hrabia (92), Tadeusz (70), Wojski (69), Tadeusza (54),
     Gerwazy (51), Robak (42), Podkomorzy (41), Klucznik (40), Telimena (38).
"""
