import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# --------------------------------------------------
# Wczytanie tekstu dramatu Hamlet
# --------------------------------------------------

with open("texts/hamlet-william-shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --------------------------------------------------
# Wybór 5 zdań z właściwego tekstu dramatu
# (pomijamy metadane, notatki edytorskie i licencję Project Gutenberg)
# --------------------------------------------------

# Odszukanie początku dialogu w Akcie I
dialogue_start = text.find("Nay, answer me")
if dialogue_start == -1:
    dialogue_start = text.find("ACT I")

dialogue_text = text[dialogue_start:]
raw_sentences = sent_tokenize(dialogue_text)

sentences = []
for s in raw_sentences:
    # Oczyszczenie z didaskaliów w nawiasach i oznaczeń postaci
    cleaned = re.sub(r"_[A-Za-z\.]+_|\[.*?\]|\([A-Z\.]+\)", "", s)
    cleaned = " ".join(cleaned.split())
    # Filtrujemy nagłówki scen i puste linie
    if len(cleaned) > 25 and not cleaned.isupper() and not cleaned.startswith("Enter"):
        sentences.append(cleaned)
    if len(sentences) == 5:
        break

print("=" * 60)
print("WYBRANE ZDANIA Z HAMLETA (AKT I, SCENA 1):")
print("=" * 60)
for i, s in enumerate(sentences, start=1):
    print(f"{i}. {s}")

# --------------------------------------------------
# Inicjalizacja algorytmów
# Stemming: SnowballStemmer (ulepszony algorytm Portera)
# Lematyzacja: WordNetLemmatizer (oparty na bazie leksykalnej WordNet)
# --------------------------------------------------

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# WordNetLemmatizer bez podanej części mowy zakłada RZECZOWNIK ('am' -> 'am').
# Dlatego dodatkowo tagujemy zdanie (nltk.pos_tag) i przekazujemy część mowy.
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

def wordnet_pos(treebank_tag):
    return {"J": "a", "V": "v", "R": "r"}.get(treebank_tag[:1], "n")

different_results = []

for number, sentence in enumerate(sentences, start=1):
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    print(f"\n\nZDANIE NUMER {number}: \"{sentence}\"")

    h1, h2, h3, h4 = "TOKEN", "RDZEŃ (STEM)", "LEMAT (domyślny)", "LEMAT (z POS)"
    print("=" * 76)
    print(f"|{h1:^17} | {h2:^17} | {h3:^17} | {h4:^17}|")
    print("=" * 76)

    for token, tag in tagged:
        # Pomijanie znaków interpunkcyjnych
        if not token.isalpha():
            continue

        token_lower = token.lower()
        stem = stemmer.stem(token_lower)
        lemma = lemmatizer.lemmatize(token_lower)
        lemma_pos = lemmatizer.lemmatize(token_lower, wordnet_pos(tag))

        print(f"|{token:^17} | {stem:^17} | {lemma:^17} | {lemma_pos + ' (' + tag + ')':^17}|")
        print("-" * 76)

        if stem != lemma_pos or stem != lemma:
            different_results.append((token, stem, lemma_pos))

# --------------------------------------------------
# Zestawienie słów o różnych wynikach (Punkt 4)
# --------------------------------------------------

print("\n" + "=" * 65)
print("SŁOWA, DLA KTÓRYCH STEMMING I LEMATYZACJA DAŁY RÓŻNE WYNIKI:")
print("=" * 65)

# Usunięcie ewentualnych duplikatów z zachowaniem kolejności
unique_diffs = []
seen = set()
for item in different_results:
    if item[0].lower() not in seen:
        seen.add(item[0].lower())
        unique_diffs.append(item)

h1, h2, h3 = "SŁOWO (TOKEN)", "RDZEŃ (STEM)", "LEMAT (LEMMA)"
print(f"|{h1:^19} | {h2:^19} | {h3:^19}|")
print("-" * 65)
for token, stem, lemma in unique_diffs:
    print(f"|{token:^19} | {stem:^19} | {lemma:^19}|")
print("=" * 65)

# ==============================================================================
# ODPOWIEDŹ I WYJAŚNIENIE DO PUNKTU 4
# ==============================================================================
"""
Wyjaśnienie różnic dla wybranych słów:

1. 'carefully' -> rdzeń: 'care', lemat: 'carefully'
   - Stemmer Snowball odcina regułowo przyrostki przysłówkowe ('-fully' / '-ly'),
     redukując wyraz do podstawy rzeczownikowo-czasownikowej 'care'.
   - Lematyzator WordNet zachowuje formę 'carefully', ponieważ przysłówek
     stanowi osobną, pełnoprawną jednostkę leksykalną (hasło słownikowe).

2. 'thanks' -> rdzeń: 'thank', lemat: 'thanks'
   - Stemmer odcina końcówkę fleksyjną '-s', traktując wyraz jako formę czasownika
     'to thank' w 3. os. l. poj. lub liczbę mnogą rzeczownika.
   - Lematyzator WordNet traktuje 'thanks' jako samodzielny rzeczownik (plurale tantum)
     oznaczający podziękowania / dzięki, dlatego lemat pozostaje w formie 'thanks'.

3. 'haste' -> rdzeń: 'hast', lemat: 'haste'
   - Stemmer stosuje regułę usuwania końcowego niemego '-e' w celu normalizacji
     form takich jak 'hasting' czy 'hasty', co skutkuje powstaniem nieistniejącego
     w języku angielskim rdzenia 'hast'.
   - Lematyzator weryfikuje słowo w słowniku WordNet i zachowuje poprawną formę
     hasłową (mianownik) 'haste'.

4. 'struck' -> rdzeń: 'struck', lemat z POS: 'strike'   oraz   'am' -> rdzeń: 'am', lemat z POS: 'be'
   - Formy nieregularne: stemmer nie ma czego obciąć, a lematyzator ze słownikiem WordNet
     zna odmianę nieregularną – ale TYLKO gdy dostanie część mowy (czasownik). Bez POS
     WordNetLemmatizer zakłada rzeczownik i zwraca 'struck' / 'am' bez zmian.

5. ''Tis' -> lemat: 'ti' – błąd lematyzatora. Archaiczny skrót 'it is' został uznany za
   liczbę mnogą rzeczownika 'ti' (nuta w solmizacji). Tagger POS uczony na współczesnym
   angielskim myli się też przy szekspirowskiej składni (np. 'answer' oznaczone jako IN).

Podsumowanie teoretyczne:
- Stemming działa algorytmicznie (metodą heurystycznego obcinania prefiksów i sufiksów)
  bez znajomości słownika i kontekstu gramatycznego. Może prowadzić do nadmiernego
  obcinania (overstemming) i tworzenia wyrazów niegramatycznych.
- Lematyzacja korzysta z pełnej bazy morfologiczno-słownikowej (WordNet) i zawsze
  zwraca poprawny wyraz hasłowy (lemat), respektując reguły języka.
"""
