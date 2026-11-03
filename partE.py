import spacy
import morfeusz2
from nltk.stem.snowball import SnowballStemmer

# ==============================================================================
# CZĘŚĆ E: POLSZCZYZNA WYMAGA INNYCH NARZĘDZI
# ==============================================================================

print("=" * 85)
print("CZĘŚĆ E: PRZETWARZANIE JĘZYKA POLSKIEGO (spaCy, Morfeusz2 ORAZ STEMMING)")
print("=" * 85)

# --------------------------------------------------
# 1. Analiza dedykowanymi narzędziami: spaCy oraz Morfeusz2
# --------------------------------------------------

print("\n--- 1. ANALIZA DEDYKOWANYMI NARZĘDZIAMI (spaCy pl_core_news_sm oraz Morfeusz2) ---")

nlp = spacy.load("pl_core_news_sm")
morfeusz = morfeusz2.Morfeusz()

fragment = (
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie. "
    "Ile cię trzeba cenić, ten tylko się dowie, kto cię stracił."
)

doc = nlp(fragment)

# Heurystyczny stemmer regułowy (odcinanie sufiksów fleksyjnych)
def simple_polish_stemmer(word: str) -> str:
    word = word.lower()
    suffixes = [
        "eście", "iście", "iśmy", "eśmy", "ałem", "ałam", "ałeś", "ałaś",
        "iłem", "iłam", "iłeś", "iłaś", "owie", "ami", "ach", "ego", "emu",
        "iem", "om", "ów", "em", "ie", "ce", "ze", "te", "ne",
        "a", "e", "y", "i", "o", "u"
    ]
    for s in suffixes:
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[:-len(s)]
    return word

h1, h2, h3, h4, h5 = "TOKEN", "LEMAT (spaCy)", "LEMAT (Morfeusz2)", "TAG MORFOSYNTAKTYCZNY", "RDZEŃ (HEUR.)"
print(f"Fragment: \"{fragment}\"\n")
print("=" * 95)
print(f"|{h1:^12} | {h2:^15} | {h3:^18} | {h4:^26} | {h5:^13}|")
print("=" * 95)

for token in doc:
    if token.is_alpha:
        # Analiza Morfeusz2 (Słownik Gramatyczny Języka Polskiego)
        morf_res = morfeusz.analyse(token.text)
        first_interp = morf_res[0][2]
        morf_lemma = first_interp[1].split(":")[0]  # usunięcie kwalifikatorów
        morf_tag = first_interp[2]

        stem = simple_polish_stemmer(token.text)
        print(f"|{token.text:^12} | {token.lemma_:^15} | {morf_lemma:^18} | {morf_tag:<26} | {stem:^13}|")
        print("-" * 95)

# --------------------------------------------------
# 2. Zastosowanie angielskiego stemmera do polskich słów
# --------------------------------------------------

print("\n" + "=" * 85)
print("--- 2. ANGIELSKI STEMMER (SnowballStemmer('english')) NA POLSKICH SŁOWACH ---")
print("=" * 85)

english_stemmer = SnowballStemmer("english")

polish_test_words = [
    "książkami",    # końcówka narzędnika l. mn. -ami
    "chłopcami",    # końcówka narzędnika l. mn. -ami
    "warszawie",    # miejscownik l. poj. -ie
    "piękny",       # przymiotnik l. poj. m. -y
    "zrobiliśmy",   # czasownik przeszły 1 os. l. mn. -iśmy
    "samochodami",  # narzędnik l. mn. -ami
    "uczniowie"     # mianownik l. mn. -owie
]

t1, t2, t3 = "SŁOWO POLSKIE", "WYNIK ANGIELSKIEGO STEMMERA", "CO SIĘ STAŁO? (ANALIZA)"
print(f"|{t1:^18} | {t2:^27} | {t3:<40}|")
print("=" * 90)

analyses = {
    "książkami": "Brak reakcji - końcówka '-ami' nie istnieje w angielskim",
    "chłopcami": "Brak reakcji - końcówka '-ami' nie istnieje w angielskim",
    "warszawie": "Obcięto '-e' do 'warszawi' (angielska reguła 'e')",
    "piękny": "Zamiana 'y' -> 'i' ('piękni') - tworzy l. mn. zamiast rdzenia!",
    "zrobiliśmy": "Zamiana 'y' -> 'i' ('zrobiliśmi') - brak usunięcia końcówki",
    "samochodami": "Brak reakcji - końcówka '-ami' zignorowana",
    "uczniowie": "Obcięto '-e' do 'uczniowi' (tworzy celownik l. poj.!)"
}

for w in polish_test_words:
    res = english_stemmer.stem(w)
    note = analyses[w]
    print(f"|{w:^18} | {res:^27} | {note:<40}|")
    print("-" * 90)

# --------------------------------------------------
# 3. Języki obsługiwane przez SnowballStemmer
# --------------------------------------------------

print("\n" + "=" * 85)
print("--- 3. JĘZYKI OBSŁUGIWANE PRZEZ SnowballStemmer ---")
print("=" * 85)

supported_languages = sorted(list(SnowballStemmer.languages))
print(f"Liczba obsługiwanych języków: {len(supported_languages)}")
print("Lista języków:")
for i, lang in enumerate(supported_languages, 1):
    print(f"  {i:2d}. {lang}")

print("\nKomentarz dotyczący obecności lub braku języka polskiego:")
explanation = """
KOMENTARZ TEORETYCZNY:
1. Brak języka polskiego:
   SnowballStemmer obsługuje 16 języków (w tym tylko jeden słowiański: rosyjski).
   Język polski NIE JEST obsługiwany.

2. Dlaczego mechaniczny stemming zawodzi w języku polskim?
   - Bogata fleksja i koniugacja: Polski posiada 7 przypadków deklinacji, liczne rodzaje
     gramatyczne i skomplikowaną koniugację. Końcówki fleksyjne są bardzo różnorodne.
   - Oboczności rdzeniowe (alternacje): W polszczyźnie odmiana słowa wiąże się ze zmianami
     wewnątrz samego rdzenia, a nie tylko na jego końcu. Przykłady:
     * 'ręka' -> 'rąk' -> 'ręce' (wymiana ę:ą oraz k:c)
     * 'miasto' -> 'mieście' (wymiana a:e oraz st:ść)
     * 'wiózł' -> 'wieźć' -> 'wozi' (wymiana ó:e:o oraz zł:ź:z)
     * 'pies' -> 'psa' (ruchome 'e')
   - Heurystyka odcinania sufiksów: Algorytmy typu Porter / Snowball opierają się na
     mechanicznym odcinaniu końcówek z prawej strony wyrazu. Przy obocznościach rdzeniowych
     dwie formy tego samego słowa ('ręka' i 'rąk') nigdy nie zbiegną się do jednego rdzenia.

3. Rola dedykowanych narzędzi morfologicznych (spaCy i Morfeusz2):
   - W języku polskim standardem jest pełna lematyzacja słownikowo-morfologiczna.
   - Morfeusz2 (oparty na Słowniku Gramatycznym Języka Polskiego - SGJP) pozwala na
     precyzyjne rozpoznanie formy wyjściowej i przypadków (np. 'Litwo' -> mianownik 'Litwa',
     wołacz; 'Ojczyzno' -> mianownik 'ojczyzna').
   - spaCy z modelem pl_core_news_sm radzi sobie znakomicie z ujednoznacznianiem
     kontekstowym (np. 'jesteś' -> 'być', 'cię' -> 'ty').
"""
print(explanation)
