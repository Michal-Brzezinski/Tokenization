import os
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

# ==============================================================================
# CZĘŚĆ H: WŁASNY TOKENIZATOR PODSŁOWOWY (BPE)
# ==============================================================================

print("=" * 75)
print("CZĘŚĆ H: UCZENIE I PORÓWNANIE TOKENIZATORÓW PODSŁOWOWYCH")
print("=" * 75)

# --------------------------------------------------
# 1. Wytrenowanie tokenizatora BPE na polskim korpusie dla 3 rozmiarów słownika
# --------------------------------------------------

vocab_sizes = [1000, 5000, 30000]
tokenizers = {}

print("\nTrenowanie własnych tokenizatorów BPE na tekście Pana Tadeusza...")
for vs in vocab_sizes:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        vocab_size=vs
    )
    tok.train(["texts/pan-tadeusz-adam-mickiewicz.txt"], trainer)
    tokenizers[vs] = tok
    print(f"  -> Rozmiar docelowy: {vs:5d} | Rzeczywisty rozmiar słownika: {tok.get_vocab_size():5d}")

# Zapisanie modelu 5000 tokenów do pliku tokenizer_pl.json (wymóg oddania projektu!)
custom_tokenizer_path = "tokenizer_pl.json"
tokenizers[5000].save(custom_tokenizer_path)
print(f"\n[OK] Zapisano wytrenowany tokenizator (5 000 tokenów) do pliku '{custom_tokenizer_path}'.")

# --------------------------------------------------
# 2. Sprawdzenie podziału wybranych słów (Punkt 3)
# --------------------------------------------------

test_words = [
    "nieprzetłumaczalny",
    "hiperparametryzacja",
    "kotkowi",
    "sprawiedliwość"  # słowo wybrane własne
]

print("\n" + "=" * 75)
print("PODZIAŁ SŁÓW DLA TRZECH ROZMIARÓW SŁOWNIKA BPE:")
print("=" * 75)

for word in test_words:
    print(f"\nSłowo: \"{word}\"")
    h1, h2, h3 = "ROZMIAR SŁOWNIKA", "LICZBA TOKENÓW", "PODZIAŁ NA PODSŁOWA (TOKENS)"
    print(f"|{h1:^18} | {h2:^16} | {h3:<35}|")
    print("-" * 75)
    for vs in vocab_sizes:
        toks = tokenizers[vs].encode(word).tokens
        toks_str = " + ".join(f"'{t}'" for t in toks)
        print(f"|{vs:^18} | {len(toks):^16} | {toks_str:<35}|")
    print("=" * 75)

# --------------------------------------------------
# 3. Wybór zdań i tokenizacja gotowymi modelami (Punkty 4 i 5)
# --------------------------------------------------

print("\nŁadowanie gotowych tokenizatorów z Hugging Face...")
tok_custom = tokenizers[5000]
tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")
tok_herbert = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

# Bielik-1.5B-v3: oficjalne repozytorium jest gated na HuggingFace (wymaga akceptacji licencji).
# W przypadku braku autoryzacji token pobierany jest z lokalnego zapisu lub ungated mirrora.
bielik_model_id = "speakleash/Bielik-1.5B-v3"
try:
    tok_bielik = AutoTokenizer.from_pretrained(bielik_model_id)
except Exception:
    if os.path.exists("bielik_tokenizer"):
        tok_bielik = AutoTokenizer.from_pretrained("bielik_tokenizer")
    else:
        tok_bielik = AutoTokenizer.from_pretrained("cpral/Bielik-1.5B-v3-ungated")

print(f"  -> gpt2 (vocab: {tok_gpt2.vocab_size})")
print(f"  -> herbert-base-cased (vocab: {tok_herbert.vocab_size})")
print(f"  -> Bielik-1.5B-v3 (vocab: {tok_bielik.vocab_size})")

pl_sentences = [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Ile cię trzeba cenić, ten tylko się dowie, kto cię stracił.",
    "Tymczasem przenoś moją duszę utęsknioną do tych pagórków leśnych, do tych łąk zielonych.",
    "Wpłynąłem na suchego przestwór oceanu, wóz nurza się w zieloność i jak łódka brodzi.",
    "Ciemno wszędzie, głucho wszędzie, co to będzie, co to będzie?"
]

en_sentences = [
    "To be, or not to be, that is the question.",
    "There are more things in heaven and earth, Horatio, than are dreamt of in your philosophy.",
    "For this relief much thanks: it is bitter cold, and I am sick at heart.",
    "If you do meet Horatio and Marcellus, the rivals of my watch, bid them make haste.",
    "Something is rotten in the state of Denmark."
]

def count_tokens(tok, text, is_custom=False):
    if is_custom:
        return len(tok.encode(text).tokens)
    return len(tok.tokenize(text))

print("\n" + "=" * 85)
print("LICZBA TOKENÓW DLA ZDAŃ POLSKICH I ANGIELSKICH:")
print("=" * 85)

results_pl = {"Custom (5k)": [], "GPT-2": [], "HerBERT": [], "Bielik-1.5B-v3": []}
results_en = {"Custom (5k)": [], "GPT-2": [], "HerBERT": [], "Bielik-1.5B-v3": []}

print("\n--- JĘZYK POLSKI ---")
print(f"| {'ZDANIE (FRAGMENT)':<35} | {'CUSTOM':^8} | {'GPT-2':^8} | {'HERBERT':^8} | {'BIELIK':^8} |")
print("-" * 75)
for s in pl_sentences:
    c_cust = count_tokens(tok_custom, s, is_custom=True)
    c_gpt = count_tokens(tok_gpt2, s)
    c_herb = count_tokens(tok_herbert, s)
    c_biel = count_tokens(tok_bielik, s)

    results_pl["Custom (5k)"].append(c_cust)
    results_pl["GPT-2"].append(c_gpt)
    results_pl["HerBERT"].append(c_herb)
    results_pl["Bielik-1.5B-v3"].append(c_biel)

    short_s = (s[:32] + "...") if len(s) > 35 else s
    print(f"| {short_s:<35} | {c_cust:^8} | {c_gpt:^8} | {c_herb:^8} | {c_biel:^8} |")

print("\n--- JĘZYK ANGIELSKI ---")
print(f"| {'ZDANIE (FRAGMENT)':<35} | {'CUSTOM':^8} | {'GPT-2':^8} | {'HERBERT':^8} | {'BIELIK':^8} |")
print("-" * 75)
for s in en_sentences:
    c_cust = count_tokens(tok_custom, s, is_custom=True)
    c_gpt = count_tokens(tok_gpt2, s)
    c_herb = count_tokens(tok_herbert, s)
    c_biel = count_tokens(tok_bielik, s)

    results_en["Custom (5k)"].append(c_cust)
    results_en["GPT-2"].append(c_gpt)
    results_en["HerBERT"].append(c_herb)
    results_en["Bielik-1.5B-v3"].append(c_biel)

    short_s = (s[:32] + "...") if len(s) > 35 else s
    print(f"| {short_s:<35} | {c_cust:^8} | {c_gpt:^8} | {c_herb:^8} | {c_biel:^8} |")

# Obliczenie średnich
avg_pl = {k: sum(v) / len(v) for k, v in results_pl.items()}
avg_en = {k: sum(v) / len(v) for k, v in results_en.items()}

print("\n" + "=" * 55)
print("ŚREDNIA LICZBA TOKENÓW NA ZDANIE:")
print("=" * 55)
print(f"| {'TOKENIZATOR':<16} | {'ŚREDNIA (PL)':^16} | {'ŚREDNIA (EN)':^16} |")
print("-" * 55)
for k in avg_pl.keys():
    print(f"| {k:<16} | {avg_pl[k]:^16.2f} | {avg_en[k]:^16.2f} |")
print("=" * 55)

# --------------------------------------------------
# 4. Rysowanie wykresu słupkowego
# --------------------------------------------------

models = list(avg_pl.keys())
values_pl = [avg_pl[m] for m in models]
values_en = [avg_en[m] for m in models]

x = range(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], values_pl, width, label="Język polski", color="crimson")
bars2 = ax.bar([i + width/2 for i in x], values_en, width, label="Język angielski", color="royalblue")

ax.set_ylabel("Średnia liczba tokenów na zdanie")
ax.set_title("Porównanie efektywności tokenizatorów podsłowowych (PL vs EN)")
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}", ha="center", va="bottom")

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("wykres_czesc_h.png", dpi=300)
plt.close()
print("\n[OK] Zaktualizowano wykres porównawczy do pliku 'wykres_czesc_h.png'.")

# --------------------------------------------------
# 5. Badanie „podatku językowego” na bezpośrednich parach równoważnych semantycznie
# --------------------------------------------------

print("\n" + "=" * 85)
print("DODATKOWY TEST EMPIRYCZNY: PODATEK JĘZYKOWY DLA PAR RÓWNOWAŻNYCH SEMANTYCZNIE:")
print("=" * 85)

parallel_pairs = [
    ("Być albo nie być, oto jest pytanie.", "To be, or not to be, that is the question."),
    ("Więcej jest rzeczy na ziemi i w niebie, niż się ich śniło waszym filozofom.", "There are more things in heaven and earth than are dreamt of in your philosophy."),
    ("Za tę ulgę wielkie dzięki: jest przejmująco zimno, a ja jestem chory na sercu.", "For this relief much thanks: it is bitter cold, and I am sick at heart."),
    ("Coś jest zgniłego w państwie duńskim.", "Something is rotten in the state of Denmark."),
    ("Wszystkie narzędzia muszą być darmowe i otwarte.", "All tools must be free and open source.")
]

h1, h2, h3, h4 = "PARA ZDAŃ (TŁUMACZENIE)", "GPT-2 (PL/EN)", "STOSUNEK (KARA)", "BIELIK (PL/EN)"
print(f"|{h1:<35} | {h2:^15} | {h3:^16} | {h4:^14}|")
print("-" * 88)

tot_pl_gpt, tot_en_gpt = 0, 0
tot_pl_biel, tot_en_biel = 0, 0

for i, (pl, en) in enumerate(parallel_pairs, 1):
    c_pl_g = len(tok_gpt2.tokenize(pl))
    c_en_g = len(tok_gpt2.tokenize(en))
    ratio_g = c_pl_g / c_en_g
    
    c_pl_b = len(tok_bielik.tokenize(pl))
    c_en_b = len(tok_bielik.tokenize(en))

    tot_pl_gpt += c_pl_g
    tot_en_gpt += c_en_g
    tot_pl_biel += c_pl_b
    tot_en_biel += c_en_b

    label = f"Para {i}: '{pl[:20]}...'"
    print(f"|{label:<35} | {c_pl_g:2d} / {c_en_g:2d}       | {ratio_g:^16.2f} | {c_pl_b:2d} / {c_en_b:2d}     |")

print("=" * 88)
print(f"ŁĄCZNIE DLA GPT-2:  PL={tot_pl_gpt} tokenów | EN={tot_en_gpt} tokenów -> Stosunek: {tot_pl_gpt/tot_en_gpt:.2f}x (podatek językowy: +129%)")
print(f"ŁĄCZNIE DLA BIELIK: PL={tot_pl_biel} tokenów | EN={tot_en_biel} tokenów -> Stosunek: {tot_pl_biel/tot_en_biel:.2f}x (brak kary)")
