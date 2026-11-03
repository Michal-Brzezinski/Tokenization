import spacy
from tokenizers import Tokenizer
from stop_words import get_stop_words

# ==============================================================================
# CZĘŚĆ I: PORÓWNANIE WSZYSTKICH METOD PRZETWARZANIA TEKSTU
# ==============================================================================

print("=" * 80)
print("CZĘŚĆ I: ZESTAWIENIE METOD DLA TEGO SAMEGO FRAGMENTU TEKSTU POLSKIEGO")
print("=" * 80)

# --------------------------------------------------
# 1. Przygotowanie fragmentu tekstu (Inwokacja z Pana Tadeusza)
# --------------------------------------------------

with open("texts/pan-tadeusz-adam-mickiewicz.txt", "r", encoding="utf-8") as f:
    text = f.read()

pos_start = text.find("Litwo! Ojczyzno moja!")
pos_end = pos_start + 1850
sample_text = text[pos_start:pos_end].strip()

# Inicjalizacja narzędzi
nlp = spacy.load("pl_core_news_sm")
custom_tokenizer = Tokenizer.from_file("tokenizer_pl.json")
polish_stop_words = set(get_stop_words("polish"))

# --------------------------------------------------
# 2. Przetwarzanie czterema metodami (Punkt 1)
# --------------------------------------------------

# Wariant 1: Same wyrazy (surowe tokeny leksykalne)
doc = nlp(sample_text)
words_raw = [t.text.lower() for t in doc if t.is_alpha]

# Wariant 2: Wyrazy po lematyzacji (spaCy)
words_lemma = [t.lemma_.lower() for t in doc if t.is_alpha]

# Wariant 3: Wyrazy bez stop words
words_no_stop = [w for w in words_raw if w not in polish_stop_words]

# Wariant 4: Tokeny podsłowowe z własnego tokenizatora BPE (5 000)
clean_stream = " ".join(words_raw)
bpe_tokens = custom_tokenizer.encode(clean_stream).tokens

# --------------------------------------------------
# 3. Zestawienie w tabeli porównawczej (Punkty 1 i 2)
# --------------------------------------------------

methods_data = [
    ("1. Same wyrazy (surowe słowa)", words_raw),
    ("2. Wyrazy po lematyzacji (spaCy)", words_lemma),
    ("3. Wyrazy bez stop words (stop-words)", words_no_stop),
    ("4. Tokeny podsłowowe (Własny BPE 5k)", bpe_tokens)
]

print(f"\nAnalizowany fragment tekstu: Inwokacja (początek Księgi I, {len(sample_text)} znaków)\n")

h1, h2, h3, h4 = "METODA PRZETWARZANIA", "LICZBA JEDNOSTEK", "UNIKALNE (SŁOWNIK)", "PRZYKŁADOWY FRAGMENT"
print("=" * 95)
print(f"|{h1:^38} | {h2:^17} | {h3:^18} | {h4:<15}|")
print("=" * 95)

for name, tokens_list in methods_data:
    n_units = len(tokens_list)
    vocab_size = len(set(tokens_list))
    preview = " ".join(tokens_list[:4]) + "..."
    print(f"| {name:<36} | {n_units:^17} | {vocab_size:^18} | {preview:<15}|")
    print("-" * 95)

# --------------------------------------------------
# 4. Wnioski i komentarz teoretyczny (Punkt 3)
# --------------------------------------------------

print("\n" + "=" * 80)
print("KOMENTARZ TEORETYCZNY I PODSUMOWANIE:")
print("=" * 80)

commentary = """
1. Który wariant najmocniej zmniejsza słownik i jakim kosztem?
   - NAJMOCNIEJSZA REDUKCJA SŁOWNIKA W TEKŚCIE:
     Wariant 3 (Usunięcie stop words):
     * Rozmiar unikalnego słownika spada z 223 do 173 słów (redukcja o ~22.4%).
     * Łączna liczba jednostek w tekście spada z 279 do 178 (redukcja o ~36.2%).
     * Koszt: Całkowita destrukcja struktury gramatycznej, spójności składniowej
       i relacji logicznych w zdaniu. Odrzucenie spójników i zaimków uniemożliwia
       generowanie poprawnego języka przez modele LLM, czyniąc tę metodę przydatną
       jedynie do wyszukiwania słów kluczowych i klasyfikacji tematycznej (bag-of-words).

2. Rola i koszt lematyzacji (Wariant 2):
   - Zmniejsza słownik (z 223 do 207 lematów, redukcja o ~7.2%), zachowując w 100%
     liczbę słów i strukturę zdania (279 jednostek).
   - Koszt: Utrata precyzyjnych informacji fleksyjnych (czasu, trybu, osoby, przypadku).
     Wymaga dużego narzutu obliczeniowego (analizator morfologiczny oparty na sieciach
     neuronowych w spaCy).

3. Rola tokenizacji podsłowowej BPE (Wariant 4):
   - W obrębie pojedynczego fragmentu liczba jednostek wzrasta (z 279 do 453 tokenów),
     ponieważ rzadsze słowa są dzielone na mniejsze morfy (np. 'ojczyzno' -> 'ojczy', 'zno').
   - Zaleta strategiczna: W skali CAŁEGO JĘZYKA to BPE najskuteczniej kompresuje słownik,
     narzucając stały, z góry zdefiniowany rozmiar (np. 5 000 lub 32 000 tokenów),
     jednocześnie całkowicie eliminując problem słów nieznanych (OOV - Out of Vocabulary).
     Dlatego właśnie BPE jest standardem we współczesnych modelach językowych.
"""

print(commentary)
