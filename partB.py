import nltk

# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# --------------------------------------------------
# Wczytanie tekstu
# --------------------------------------------------

with open("texts/hamlet-william-shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --------------------------------------------------
# Pierwsze 5 zdań
# --------------------------------------------------

sentences = sent_tokenize(text)[:5]

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

print("WYBRANE ZDANIA:\n")

for i, s in enumerate(sentences, start=1):
    print(f"{i}. {s}")

for number, sentence in enumerate(sentences, start=1):

    tokens = word_tokenize(sentence)

    print(f"\n\nZDANIE NUMER {number}:")

    h1, h2, h3 = "TOKEN", "RDZEŃ", "LEMAT"
    print("=" * 53)
    print(f"|{h1:^15} | {h2:^15} | {h3:^15}|")
    print("=" * 53)

    for token in tokens:

        # pomijanie znaków interpunkcyjnych
        if not token.isalpha():
            continue

        stem = stemmer.stem(token.lower())
        lemma = lemmatizer.lemmatize(token.lower())

        print(f"|{token:^15} | {stem:^15} | {lemma:^15}|")
        print("-" * 53)