"""
Byte-Pair Encoding -- for programmers, not researchers.

Run it:      python3 tokenization_from_scratch.py
Debug it:    set a breakpoint in train_bpe and watch the merge list grow.

No torch. No autograd. No downloads. BPE is small enough to write completely,
so this file IS the tokeniser: trainer, encoder, byte-level fallback, and the
demos that show why your Hindi prompts cost three times more than your English
ones.

Papers:
  Sennrich, Haddow & Birch 2015, arXiv 1508.07909  (BPE for translation)
  Kudo & Richardson 2018,        arXiv 1808.06226  (SentencePiece)
  Radford et al. 2019, GPT-2                       (byte-level BPE)
"""

import re
from collections import Counter

import numpy as np

np.random.seed(0)


# ---------------------------------------------------------------------------
# STAGE 0 -- the byte alphabet (GPT-2's trick).
#
# BPE wants to merge SYMBOLS. If a symbol is a character, an unseen character
# has no id and you need an <unk>. If a symbol is a BYTE, there are exactly 256
# of them and every possible input is encodable. Forever. No <unk> ever.
#
# The catch: bytes 0-31 are control characters that would make debug output
# unreadable. So we map all 256 bytes to 256 *printable* unicode chars, one to
# one, reversibly. It is cosmetic. The alphabet is still 256 wide.
# ---------------------------------------------------------------------------
def byte_encoder():
    keep = (list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1)))
    table, spare = dict(), 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + spare)   # park it above the printable range
            spare += 1
    return table


BYTE_TO_CHAR = byte_encoder()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def to_symbols(word, byte_level=True):
    """A word -> the tuple of starting symbols BPE will merge."""
    if byte_level:
        return tuple(BYTE_TO_CHAR[b] for b in word.encode("utf-8"))
    return tuple(word)          # character-level: one symbol per character


def detokenise(tokens):
    """Bytes back to text. Lossless -- this is the whole point of byte-level."""
    raw = bytes(CHAR_TO_BYTE[c] for c in "".join(tokens))
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# STAGE 1 -- pre-tokenisation.
#
# Before BPE ever runs, the text is chopped into chunks that merges may not
# cross. GPT-2 keeps the leading space ON the word: " the" is one chunk, not
# a space plus "the". That single decision is why " the" and "the" are two
# different tokens with two different ids, and why a prompt ending in a space
# behaves oddly.
#
# SentencePiece does the same job differently: it treats the raw stream --
# spaces included -- as the input and writes the space as a visible symbol,
# so it needs no whitespace rules at all. That is why it works for Japanese
# and Thai, which do not put spaces between words.
# ---------------------------------------------------------------------------
def pre_tokenise(text):
    return re.findall(r" ?[^\s]+|\s+", text)


def word_freqs(corpus, byte_level=True, pretokenise=True):
    """corpus text -> {symbol tuple: count}. The trainer's whole input."""
    chunks = pre_tokenise(corpus) if pretokenise else [corpus]
    counts = Counter(chunks)
    return {to_symbols(w, byte_level): n for w, n in counts.items()}


# ---------------------------------------------------------------------------
# STAGE 2 -- THE WHOLE PAPER. Sennrich et al., the method section.
#
#   1. count every adjacent pair of symbols
#   2. merge the most frequent pair everywhere
#   3. repeat n times
#
# That is it. It is `Counter.most_common(1)` in a for loop. The 1994 version
# was a compression algorithm; the 2015 paper's contribution was pointing it
# at words so a translation model could spell out a word it had never seen
# instead of emitting <unk>.
# ---------------------------------------------------------------------------
def count_pairs(freqs):
    pairs = Counter()
    for symbols, n in freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += n            # weighted by how often the word occurs
    return pairs


def merge_pair(symbols, pair):
    """Glue every occurrence of `pair` in one word into a single symbol."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(freqs, n_merges, trace=0, base_size=0):
    """Return the ordered merge list. Order IS the model -- rank matters."""
    merges = []
    for step in range(n_merges):
        pairs = count_pairs(freqs)
        if not pairs:
            break
        # ties broken by the pair itself, so two runs give identical output
        best_count = max(pairs.values())
        best = min(p for p, c in pairs.items() if c == best_count)
        freqs = {merge_pair(w, best): n for w, n in freqs.items()}
        merges.append(best)
        if step < trace:
            # vocabulary = the base alphabet plus one new id per merge so far
            print(f"  merge {step + 1:2d}: {best[0]!r} + {best[1]!r}"
                  f" -> {best[0] + best[1]!r:<10} seen {best_count:>2}x,"
                  f" vocab {base_size + step + 1}")
    return merges


# ---------------------------------------------------------------------------
# STAGE 3 -- encoding. Apply the learned merges to a NEW word, in rank order.
#
# Note what is NOT here: no search, no probability, no model. Encoding is a
# deterministic replay of the training-time merge list. Which is exactly why
# a merge list from the wrong corpus fails silently -- every merge still
# applies, it just applies to the wrong places.
# ---------------------------------------------------------------------------
def encode_word(word, merges, byte_level=True):
    ranks = {pair: i for i, pair in enumerate(merges)}
    symbols = to_symbols(word, byte_level)
    while len(symbols) > 1:
        candidates = [(ranks[p], p) for p in zip(symbols, symbols[1:])
                      if p in ranks]
        if not candidates:
            break
        symbols = merge_pair(symbols, min(candidates)[1])   # lowest rank first
    return list(symbols)


def encode(text, merges, byte_level=True):
    out = []
    for chunk in pre_tokenise(text):
        out.extend(encode_word(chunk, merges, byte_level))
    return out


def build_vocab(merges, byte_level=True):
    """id table: the base alphabet, then one new id per merge, in order."""
    base = sorted(BYTE_TO_CHAR.values()) if byte_level else sorted(BASE_CHARS)
    vocab = {s: i for i, s in enumerate(base)}
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    return vocab


def to_ids(tokens, vocab):
    """What the model actually receives. Not letters. Integers."""
    return [vocab.get(t, -1) for t in tokens]


# ---------------------------------------------------------------------------
# STAGE 4 -- toy corpora. Small enough that you can read the merge list and
# check it by hand, which is the only reason to trust any of this.
# ---------------------------------------------------------------------------
TINY = ("low low low low low lower lower newest newest newest "
        "newest newest newest widest widest widest")

ENGLISH = (
    "the model reads the text and the text becomes tokens . "
    "the tokeniser is trained on text , not learned by the model . "
    "a token is a piece of a word and a word is a piece of the text . "
    "training the tokeniser on english text makes english text cheap . "
    "the model never sees the letters , the model only sees the ids . "
    "tokens are ids and ids are what the model reads , nothing else . "
    "a strawberry is a berry and a raspberry is a berry too . "
    "berry after berry after berry , the tokeniser learns berry . "
) * 6

BASE_CHARS = sorted(set(TINY))

SENTENCES = {
    "English": "The cat sat on the mat and watched the rain .",
    "Tamil":   "பூனை பாயில் அமர்ந்து மழையைப் பார்த்தது .",
    "Hindi":   "बिल्ली चटाई पर बैठकर बारिश देख रही थी ।",
}


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def show(tokens):
    """Print tokens the way a tokeniser debugger does: pipe-separated."""
    return "|".join(t.replace("Ġ", "_").replace(" ", "_") for t in tokens)


def demo_1_train_it():
    """Count pairs, merge the most frequent, repeat. Watch the vocab grow."""
    line("DEMO 1: training BPE is a Counter in a for loop")

    print("corpus (character-level, no byte tricks, so you can read it):")
    print(" ", TINY)
    freqs = word_freqs(TINY, byte_level=False)
    start = {s for w in freqs for s in w}
    print(f"\nstarting vocabulary: {len(start)} characters"
          f" -> {''.join(sorted(start))!r}")
    print("\nmerges, in the order they were learned:")
    merges = train_bpe(freqs, 12, trace=12, base_size=len(start))

    print("\nthe merge list IS the model. 12 rules, learned by counting.")
    print("11 characters in, 23 ids out, and ' newest' is now a single one.")
    return merges


def demo_2_encode(merges):
    """Common word -> one token. Rare word -> several. Unseen word -> pieces."""
    line("DEMO 2: common words are cheap, rare words are spelled out")

    print(f"{'word':>12}  {'tokens':<34} count")
    print(f"{'-' * 12:>12}  {'-' * 34:<34} -----")
    for word in [" low", " lower", " newest", " widest",
                 " lowest", " slowness", " kubernetes"]:
        toks = encode_word(word, merges, byte_level=False)
        print(f"{word.strip():>12}  {show(toks):<34} {len(toks)}")

    print("\n' newest' appeared 6 times, so it is ONE token. ' lower' only")
    print("twice, so it is three: the frequent stem plus its leftovers.")
    print("'lowest' was NEVER in the corpus, but ' low' and 'est' both were,")
    print("so it costs 2 tokens instead of 7. That is the 2015 paper's point:")
    print("open vocabulary. A word you never trained on is still spellable.")
    print("' kubernetes' shares almost nothing with the corpus, so it falls")
    print("apart into characters -- 10 tokens. Correct, just expensive.")


def demo_3_character_blindness(merges_en, vocab_en):
    """Why the model cannot count the r's in strawberry."""
    line("DEMO 3: the model cannot see letters")

    for word in [" strawberry", " raspberry"]:
        toks = encode(word, merges_en)
        ids = to_ids(toks, vocab_en)
        print(f"{word.strip():>12}: {show(toks)}")
        print(f"{'ids':>12}: {ids}")
        plural = "" if len(toks) == 1 else "s"
        print(f"{'letters':>12}: {len(word.strip())} characters ->"
              f" {len(toks)} token{plural}\n")

    toks = encode(" strawberry", merges_en)
    print("the model is asked 'how many r in strawberry'. What it received:")
    print("  ", to_ids(toks, vocab_en))
    print("One integer. There is no 'r' anywhere in what the model received.")
    print("Counting the r's means counting inside a symbol it only ever saw")
    print("as one opaque id -- like being asked how many times the letter e")
    print("appears in file handle 361.")
    print("\nHonest caveat: models often get it right anyway, because spelling")
    print("is discussed in the training text and the fact can be memorised.")
    print("Tokenisation is the mechanical reason it is HARD, not a proof that")
    print("it is impossible.")


def demo_4_multilingual_tax(merges_en):
    """The same sentence, three scripts, one English-trained tokeniser."""
    line("DEMO 4: the multilingual tax")

    base = None
    print(f"{'language':>9}  {'chars':>5}  {'tokens':>6}  {'ratio':>6}")
    for name, text in SENTENCES.items():
        toks = encode(text, merges_en)
        if base is None:
            base = len(toks)
        print(f"{name:>9}  {len(text):>5}  {len(toks):>6}"
              f"  {len(toks) / base:>5.2f}x")

    print("\nSame sentence. Same tokeniser. Trained on English text.")
    print()
    for name in ("English", "Tamil", "Hindi"):
        text = SENTENCES[name]
        n_chars = len(text.replace(" ", ""))
        n_toks = len(encode(text, merges_en))
        print(f"{name:>9}: {n_toks / n_chars:>4.2f} tokens per character")
    print("\nEvery Tamil character is 3 UTF-8 bytes and no merge ever learned")
    print("them, so each character costs about 3 tokens. Your context window")
    print("shrinks by that ratio and your per-token bill grows by it. This is")
    print("a property of the tokeniser, not of the language.")


def demo_5_leading_space(merges_en, vocab_en):
    """'the' and ' the' are different tokens. Print both ids."""
    line("DEMO 5: the leading space is part of the token")

    for word in ["the", " the"]:
        toks = encode_word(word, merges_en)
        print(f"{word!r:>7} -> {show(toks):<12} ids {to_ids(toks, vocab_en)}")

    same = encode_word("the", merges_en) == encode_word(" the", merges_en)
    print(f"\nsame tokens? {same}")
    print("Pre-tokenisation glues the space onto the following word, so the")
    print("model learned ' the' (mid-sentence) and 'the' (start of line) as")
    print("two separate things. End your prompt with a trailing space and you")
    print("hand the model a token boundary it almost never saw in training.")


def demo_6_byte_fallback(merges_en, vocab_en):
    """Anything encodes. Emoji, unseen scripts, raw bytes. No <unk>."""
    line("DEMO 6: byte-level BPE cannot fail")

    for text in ["🍓", "வணக்கம்", "\x00\x01"]:
        toks = encode(text, merges_en)
        ids = to_ids(toks, vocab_en)
        label = repr(text)
        print(f"{label:>14} -> {len(text.encode('utf-8')):>2} utf-8 bytes,"
              f" {len(toks):>2} tokens, unknown ids: {ids.count(-1)}")
        print(f"{'round trip':>14} -> {detokenise(toks)!r}")

    toks = encode("🍓", merges_en)
    print(f"\none strawberry emoji = {len(toks)} tokens:"
          f" {to_ids(toks, vocab_en)}")
    print("Four bytes, four ids, none of them meaning 'strawberry'. This is")
    print("why one emoji in the wrong place breaks a JSON schema you thought")
    print("was safe: the model has to emit four correct tokens in a row to")
    print("produce one character.")


def demo_7_break_it(merges_en, vocab_en):
    """Five deliberate breaks, printed next to the working version."""
    line("DEMO 7: break it on purpose")

    sample = "the tokeniser is trained on text"

    print("--- BREAK 1: vocabulary too small ---")
    freqs = word_freqs(ENGLISH)
    for n in (0, 20, 100, 400):
        m = train_bpe(freqs, n)
        toks = encode(sample, m)
        print(f"  {n:>3} merges -> {len(toks):>3} tokens for a"
              f" {len(sample)}-character sentence")
    print("  0 merges is pure bytes. Every merge you drop, you pay for at")
    print("  inference time, forever, on every request.")

    print("\n--- BREAK 2: drop the byte-level fallback ---")
    char_merges = train_bpe(word_freqs(ENGLISH, byte_level=False), 200)
    ok = encode_word(" text", char_merges, byte_level=False)
    print(f"  character-level, english word:  {show(ok)}")
    try:
        toks = encode_word("பூனை", char_merges, byte_level=False)
        vocab = {s for pair in char_merges for s in pair}
        missing = [t for t in toks if t not in vocab]
        print(f"  character-level, tamil word:    {show(toks)}")
        print(f"  symbols with no id at all:      {len(missing)} ->"
              f" every one becomes <unk>")
    except KeyError as e:
        print(f"  character-level, tamil word:    KeyError {e}")
    print(f"  byte-level, same tamil word:    "
          f"{len(encode_word('பூனை', merges_en))} tokens, 0 unknown")

    print("\n--- BREAK 3: skip pre-tokenisation ---")
    m_nopre = train_bpe(word_freqs(ENGLISH, pretokenise=False), 300)
    crossers = [a + b for a, b in m_nopre if " " in (a + b).replace("Ġ", " ")
                or "Ġ" in a + b]
    spanning = [t for t in
                (a + b for a, b in m_nopre)
                if t.count("Ġ") > 1]
    print(f"  merges learned: {len(m_nopre)}")
    print(f"  merges containing a space: {len(crossers)}")
    print(f"  tokens spanning TWO words: {len(spanning)},"
          f" e.g. {show(spanning[:3])}")
    print("  A single id now means 'the model' as a unit. Say 'the' followed")
    print("  by anything else and the tokeniser has to fall back to shorter,")
    print("  rarer pieces the model saw far less often.")

    print("\n--- BREAK 4: encode with somebody else's merge list ---")
    tamil_merges = train_bpe(word_freqs(SENTENCES["Tamil"] * 20), 300)
    right = encode(sample, merges_en)
    wrong = encode(sample, tamil_merges)
    print(f"  right merge list: {len(right):>3} tokens  {show(right[:8])} ...")
    print(f"  wrong merge list: {len(wrong):>3} tokens  {show(wrong[:8])} ...")
    print("  No error. No warning. Valid ids the whole way. Just silently")
    print("  worse splits, and a model that never saw these ids together.")

    print("\n--- BREAK 5: strip the leading space on a continuation ---")
    with_space = encode(" continues the sentence", merges_en)
    without = encode("continues the sentence", merges_en)
    print(f"  ' continues...' -> {len(with_space)} tokens,"
          f" first id {to_ids(with_space, vocab_en)[0]}")
    print(f"  'continues...'  -> {len(without)} tokens,"
          f" first id {to_ids(without, vocab_en)[0]}")
    print(f"  identical? {with_space == without}")


if __name__ == "__main__":
    tiny_merges = demo_1_train_it()
    demo_2_encode(tiny_merges)

    # The English byte-level tokeniser used by every demo from here down.
    merges_en = train_bpe(word_freqs(ENGLISH), 400)
    vocab_en = build_vocab(merges_en)
    print(f"\n[byte-level tokeniser: 256 byte ids + {len(merges_en)} merges"
          f" = {len(vocab_en)} ids]")

    demo_3_character_blindness(merges_en, vocab_en)
    demo_4_multilingual_tax(merges_en)
    demo_5_leading_space(merges_en, vocab_en)
    demo_6_byte_fallback(merges_en, vocab_en)
    demo_7_break_it(merges_en, vocab_en)
    print()
