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
    text = f.read()

# --------------------------------------------------
# Czyszczenie tekstu
# --------------------------------------------------

# małe litery
text = text.lower()

# usunięcie znaków specjalnych
text = re.sub(r"[^\w\s]", " ", text)

# zwinięcie wielokrotnych białych znaków
text = re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------
# Tokenizacja
# --------------------------------------------------

words = word_tokenize(text, language="english")
sentences = sent_tokenize(text)

print(f"Liczba zdań: {len(sentences)}")
print(f"Liczba wyrazów: {len(words)}")

# --------------------------------------------------
# Generowanie n-gramów
# --------------------------------------------------

unigrams = list(ngrams(words, 1))
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))

print(f"Unigramy: {len(unigrams)}")
print(f"Bigramy: {len(bigrams)}")
print(f"Trigramy: {len(trigrams)}")

# --------------------------------------------------
# Liczenie częstości
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
# Histogramy n-gramów
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --------------------------------------------------
# Unigramy
# --------------------------------------------------

labels_ug = [unigram[0] for unigram, _ in top_unigrams]
values_ug = [count for _, count in top_unigrams]

axes[0].bar(labels_ug, values_ug)
axes[0].set_title("10 najczęstszych unigramów")
axes[0].tick_params(axis="x", rotation=45)

# --------------------------------------------------
# Bigramy
# --------------------------------------------------

labels_bg = [" ".join(bigram) for bigram, _ in top_bigrams]
values_bg = [count for _, count in top_bigrams]

axes[1].bar(labels_bg, values_bg)
axes[1].set_title("10 najczęstszych bigramów")
axes[1].tick_params(axis="x", rotation=45)

# --------------------------------------------------
# Trigramy
# --------------------------------------------------

labels_tg = [" ".join(trigram) for trigram, _ in top_trigrams]
values_tg = [count for _, count in top_trigrams]

axes[2].bar(labels_tg, values_tg)
axes[2].set_title("10 najczęstszych trigramów")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()


# ------------------  WNIOSKI ---------------------

