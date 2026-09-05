"""
Retrieval-Augmented Generation -- for programmers, not researchers.

Run it:      python3 rag_from_scratch.py
Debug it:    set a breakpoint in retrieve() and step through one question.

No torch. No downloads. No language model. The "generation" step here builds
the prompt and prints it, because the prompt is the only part of the generator
you control anyway. Everything else is the retriever -- the half you build,
the half that breaks, the half this script is about.
"""

import re
import textwrap
import zlib
import numpy as np

np.random.seed(0)            # nothing in here is random; the seed is site policy
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the corpus. One internal wiki page about a made-up job scheduler
# called Lantern. Invented, so nothing here can be checked against the world;
# internally consistent, so every question has one right answer.
#
# The first sentence is the README's opening paragraph: it mentions everything
# and answers nothing. Every real corpus has one. Keep an eye on it.
#
# In the paper the corpus is Wikipedia, cut into 21M chunks of 100 words.
# Same idea, more rows.
# ---------------------------------------------------------------------------
DOCUMENT = (
    "Lantern is the internal job scheduler at Fennel Systems: every batch job "
    "at Fennel is submitted to Lantern, queued by Lantern, given a timeout by "
    "Lantern, and retried by Lantern when it fails; failed jobs that keep "
    "failing are kept by Lantern in the dead-letter table, and failed jobs "
    "stay there until they expire. "
    "Lantern was first deployed in March 2019. "
    "Lantern stores its job queue in Postgres. "
    "The Postgres cluster behind Lantern runs on three nodes. "
    "Lantern's default job timeout is 45 minutes. "
    "A job that exceeds its timeout is retried at most twice. "
    "Lantern workers pull jobs over gRPC on port 7020. "
    "Lantern exports its metrics to Prometheus every 15 seconds. "
    "The Lantern dashboard is served by a separate service called Beacon. "
    "Beacon is written in Go and listens on port 8080. "
    "Beacon reads job state from a Redis replica, not from Postgres. "
    "The Redis replica behind Beacon lags Postgres by about two seconds. "
    "Lantern is owned by the Platform team. "
    "The Platform team's on-call rotation is one week long. "
    "Priya Raman is the current tech lead of the Platform team. "
    "Lantern's config file lives at /etc/lantern/lantern.toml. "
    "Setting max_workers above 64 in lantern.toml is rejected at startup. "
    "Lantern release 3.2 added cron-style schedules. "
    "Cron-style schedules in Lantern are evaluated in UTC, never in local time. "
    "Failed jobs are kept in the dead-letter table for 30 days. "
    "Lantern has no built-in authentication; access is controlled by the VPN."
)


def chunk(document, sentences_per_chunk=1):
    """Split a document into retrievable pieces.

    The paper uses 100-word windows. Here it is one sentence per chunk, so a
    hit is readable at a glance. Chunk size is the most consequential knob in
    the whole pipeline and nobody tunes it -- demo 7 shows why they should."""
    sentences = re.split(r"(?<=\.)\s+", document.strip())
    return [" ".join(sentences[i:i + sentences_per_chunk])
            for i in range(0, len(sentences), sentences_per_chunk)]


# ---------------------------------------------------------------------------
# STAGE 1 -- tokenise. Lowercase, split on anything that isn't a letter or a
# digit, drop the glue words. No stemming: "job" and "jobs" are different
# tokens, "owns" and "owned" are different tokens. That is a real weakness of
# any word-overlap retriever, and demo 3 shows it costing us.
# ---------------------------------------------------------------------------
STOPWORDS = set("a an and are at by does for from how in is it its of on or "
                "that the this to was what which who with".split())


def tokenise(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


# ---------------------------------------------------------------------------
# STAGE 2 -- "embed". The paper uses a BERT bi-encoder (DPR) that was TRAINED
# so a question and its answer passage land near each other. We have no model,
# so we use the oldest trick in search: a bag of words, weighted by tf-idf,
# hashed into a fixed-width vector, L2-normalised.
#
# Same output shape as DPR -- one fixed-size vector per chunk, compared by dot
# product -- so the pipeline built on top of it is the real pipeline. What it
# lacks is meaning: "owns" and "owned" share no bucket. DPR would know.
# ---------------------------------------------------------------------------
D = 1024   # embedding width. DPR's is 768. Ours is a hash table, not a model.


def bucket(token):
    """Hashing trick: token -> a fixed slot in [0, D). No vocabulary to build,
    store or ship. crc32 rather than hash() because hash() is salted per
    process and the index would not survive a restart."""
    return zlib.crc32(token.encode()) % D


def fit_idf(chunks):
    """Inverse document frequency, one weight per bucket.

    A word in most chunks ("lantern") is worth little; a word in one chunk
    ("timeout") is worth a lot. log((N+1)/(df+1)) + 1, the smoothed textbook
    form. Same instinct as BM25, minus the tuning knobs."""
    df = np.zeros(D)
    for c in chunks:
        for b in set(bucket(t) for t in tokenise(c)):
            df[b] += 1
    return np.log((len(chunks) + 1) / (df + 1)) + 1


def embed(text, idf, normalise=True):
    """text -> (D,) vector.

    (1) count each bucket           -- term frequency
    (2) multiply by idf             -- rare words count more
    (3) divide by the L2 norm       -- so dot product == cosine, length-blind

    Step 3 is the one people delete. Demo 5 shows what happens."""
    v = np.zeros(D)
    for t in tokenise(text):
        v[bucket(t)] += 1
    v *= idf
    if normalise:
        v /= np.linalg.norm(v) + 1e-12
    return v


# ---------------------------------------------------------------------------
# STAGE 3 -- index and retrieve. The index is a matrix with one row per chunk.
# At 21M rows that matrix is what FAISS holds for you, with an approximate
# nearest-neighbour structure (HNSW in the paper) so you don't scan every row.
# At 21 rows a NumPy array is the same object and a full scan is fine.
# ---------------------------------------------------------------------------
def build_index(chunks, idf, normalise=True):
    return np.stack([embed(c, idf, normalise) for c in chunks])


def retrieve(question, index, idf, k=3, normalise=True):
    """The whole retriever: one matrix-vector product and an argsort.

    scores[i] = similarity(question, chunk i). Return the k best, best first.
    There is no "nothing matched" -- argsort always returns k rows."""
    q = embed(question, idf, normalise)
    scores = index @ q                        # (n_chunks,)
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# STAGE 4 -- "generate". Assemble the prompt. This string is what a real
# system sends to the LLM; there is no LLM in this script, so we print it.
#
# The paper does NOT do this. RAG-Sequence runs the generator once per
# retrieved passage and averages the k answer distributions, weighted by the
# retrieval scores. Production systems paste all k passages into one prompt
# and call the model once. Same information, one call instead of k.
# ---------------------------------------------------------------------------
def build_prompt(question, passages):
    context = "\n".join(textwrap.fill(f"[{i + 1}] {p}", 78,
                                      subsequent_indent="    ")
                        for i, p in enumerate(passages))
    return ("Answer using only the context below.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"Question: {question}\nAnswer:")


# ===========================================================================
# DEMOS
# ===========================================================================
QUESTIONS = [
    # (question, index of the chunk that answers it)
    ("What is Lantern's default job timeout?", 4),
    ("Which team owns Lantern?", 12),
    ("How long are failed jobs kept in the dead-letter table?", 19),
]


def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def show_hits(hits, chunks, answer=None):
    """One row per hit. A leading * marks the chunk that answers the question."""
    for rank, (i, s) in enumerate(hits, 1):
        mark = "*" if i == answer else " "
        text = chunks[i] if len(chunks[i]) <= 58 else chunks[i][:55] + "..."
        print(f"{mark} #{rank}  {s:.3f}  chunk {i:2d}  {text}")


def demo_1_corpus_and_chunks():
    line("DEMO 1: the corpus, chunked")
    chunks = chunk(DOCUMENT)
    n0 = len(tokenise(chunks[0]))
    print(f"{len(chunks)} chunks, one sentence each\n")
    for i in (0, 4, 12, 19):
        n = len(tokenise(chunks[i]))
        print(textwrap.fill(f"chunk {i:2d}  ({n:2d} tokens)  {chunks[i]}", 78,
                            subsequent_indent=" " * 24))
    print(f"\nREAD THIS: chunk 0 is the README paragraph. {n0} tokens, mentions")
    print("timeout, failed jobs, the dead-letter table -- and answers nothing.")
    print("It is going to keep turning up.")


def demo_2_embedding():
    line("DEMO 2: a chunk becomes a vector with no model in sight")
    chunks = chunk(DOCUMENT)
    idf = fit_idf(chunks)
    c = chunks[4]
    toks = tokenise(c)
    print(f"chunk 4:  {c}")
    print(f"tokens:   {toks}")
    print(f"buckets:  {[bucket(t) for t in toks]}")
    v = embed(c, idf)
    nz = np.flatnonzero(v)
    print(f"\nvector shape: {v.shape}   nonzero entries: {len(nz)} of {D}")
    print(f"L2 norm: {np.linalg.norm(v):.3f}")
    for t in toks:
        print(f"  bucket {bucket(t):4d}  {t:8s}  idf {idf[bucket(t)]:.2f}"
              f"  weight {v[bucket(t)]:.3f}")

    vocab = set(t for c in chunks for t in tokenise(c))
    buckets = [bucket(t) for t in vocab]
    collisions = len(buckets) - len(set(buckets))
    df_lantern = sum("lantern" in tokenise(c) for c in chunks)
    print(f"\nvocabulary: {len(vocab)} distinct tokens -> {D} buckets,"
          f" {collisions} hash collisions")
    print(f"\nREAD THIS: 'lantern' is in {df_lantern} of {len(chunks)} chunks,"
          " so it gets the smallest")
    print("weight; 'default', '45' and 'minutes' are in one chunk each, so they")
    print(f"carry the vector. That is all tf-idf is. {D - len(nz)} of the"
          f" {D} slots are zero.")


def demo_3_retrieval():
    line("DEMO 3: cosine top-3 for three questions  (* = the answer chunk)")
    chunks = chunk(DOCUMENT)
    idf = fit_idf(chunks)
    index = build_index(chunks, idf)
    for q, ans in QUESTIONS:
        print(f"\nQ: {q}")
        print(f"   query tokens: {tokenise(q)}")
        show_hits(retrieve(q, index, idf, k=3), chunks, answer=ans)
    print("\nREAD THIS: all three answers land at #1. Look at question 2 though:")
    print("'owns' matched nothing -- the corpus says 'owned'. The hit came from")
    print("'team' + 'lantern' alone, scored 0.34, and the runners-up got there on")
    print("'team' by itself. A learned encoder (DPR) would score 'owns' and")
    print("'owned' as the same idea. This one can't; it has never seen a word.")


def demo_4_the_prompt():
    line("DEMO 4: the assembled prompt -- what actually goes to the LLM")
    chunks = chunk(DOCUMENT)
    idf = fit_idf(chunks)
    index = build_index(chunks, idf)
    q, _ = QUESTIONS[0]
    hits = retrieve(q, index, idf, k=3)
    prompt = build_prompt(q, [chunks[i] for i, _ in hits])
    print(prompt)
    print(f"\n({len(prompt.split())} words. An LLM call goes here. There is none"
          " in this script.)")

    print("\n--- same question, no retrieval at all ---\n")
    print(build_prompt(q, []))
    print("\nREAD THIS: with '(none)' the model answers from its weights. Lantern")
    print("is private; the weights have never seen it; the model will make up a")
    print("number and sound sure. That is the 'hallucination' everyone blames on")
    print("the LLM. It was a retrieval miss.")


def demo_5_raw_dot_product():
    line("DEMO 5: BREAK IT -- skip L2 normalisation, use the raw dot product")
    chunks = chunk(DOCUMENT)
    idf = fit_idf(chunks)
    cos_index = build_index(chunks, idf, normalise=True)
    raw_index = build_index(chunks, idf, normalise=False)
    lengths = [len(tokenise(c)) for c in chunks]
    longest = int(np.argmax(lengths))
    print(f"longest chunk is chunk {longest} at {lengths[longest]} tokens;"
          f" the median chunk is {int(np.median(lengths))} tokens\n")
    wins = 0
    for q, ans in QUESTIONS:
        c = retrieve(q, cos_index, idf, k=1)[0]
        r = retrieve(q, raw_index, idf, k=1, normalise=False)[0]
        wins += r[0] == longest
        print(f"Q: {q}")
        print(f"   cosine #1:  chunk {c[0]:2d}  ({c[1]:.3f})"
              f"      raw dot #1:  chunk {r[0]:2d}  ({r[1]:.2f})")
    print(f"\nREAD THIS: without the divide-by-norm, a long chunk has a long")
    print("vector, and a long vector wins dot products it has no business")
    print(f"winning. Chunk {longest} takes {wins} of {len(QUESTIONS)} questions,"
          " with the biggest scores on the")
    print("page. Your retriever has become 'return the longest paragraph'.")


def demo_6_answer_not_in_corpus():
    line("DEMO 6: BREAK IT -- ask something the corpus cannot answer")
    chunks = chunk(DOCUMENT)
    idf = fit_idf(chunks)
    index = build_index(chunks, idf)
    good_q, good_ans = QUESTIONS[1]
    bad_q = "Who is the tech lead of the Security team?"
    good = retrieve(good_q, index, idf, k=3)
    bad = retrieve(bad_q, index, idf, k=3)
    print(f"Q (answer IS in corpus):  {good_q}")
    show_hits(good, chunks, answer=good_ans)
    print(f"\nQ (answer NOT in corpus): {bad_q}")
    show_hits(bad, chunks)
    print(f"\nREAD THIS: the unanswerable question scores {bad[0][1]:.3f};"
          f" the answerable one")
    print(f"scored {good[0][1]:.3f}. top-k has no 'I don't know' -- argsort"
          " always returns k rows.")
    print("The prompt will say 'Priya Raman ... Platform team' and a model that")
    print("skims will answer 'Priya Raman'. A score threshold would have to sit")
    print(f"above {bad[0][1]:.2f} and below {good[0][1]:.2f} at the same time."
          " It can't. You need a")
    print("reranker, or a model that actually reads the context and refuses.")


def demo_7_chunk_size():
    line("DEMO 7: BREAK IT -- merge the corpus into 3 giant chunks")
    small = chunk(DOCUMENT, 1)
    big = chunk(DOCUMENT, 7)
    print(f"small: {len(small)} chunks of 1 sentence    "
          f"big: {len(big)} chunks of 7 sentences,"
          f" {[len(tokenise(c)) for c in big]} tokens")
    for n, (q, _) in enumerate(QUESTIONS, 1):
        print(f"Q{n}: {q}")
    header = (f"\n{'':6s}{'#1 score':>9s}{'#2 score':>10s}{'gap':>7s}"
              "   #1 is    answer in #1?")
    gaps, missed = {}, []
    for name, chunks in (("small", small), ("big", big)):
        idf = fit_idf(chunks)
        index = build_index(chunks, idf)
        print(f"{header}\n{name} chunks:")
        for n, (q, ans) in enumerate(QUESTIONS, 1):
            (i1, s1), (i2, s2) = retrieve(q, index, idf, k=2)
            sentence = small[ans]
            found = sentence in chunks[i1]
            where = "yes" if found else "NO"
            if found and name == "big":
                sents = re.split(r"(?<=\.)\s+", chunks[i1])
                where += f", sentence {sents.index(sentence) + 1} of {len(sents)}"
            gaps[(name, n)] = s1 - s2
            if name == "big" and not found:
                missed.append(f"Q{n}")
            print(f"  Q{n}  {s1:9.3f} {s2:9.3f} {s1 - s2:7.3f}   chunk {i1:2d}"
                  f"   {where}")

    idf_s, idf_b = fit_idf(small), fit_idf(big)
    q = QUESTIONS[0][0]
    hits_s = retrieve(q, build_index(small, idf_s), idf_s, k=3)
    hits_b = retrieve(q, build_index(big, idf_b), idf_b, k=3)
    words_s = len(build_prompt(q, [small[i] for i, _ in hits_s]).split())
    words_b = len(build_prompt(q, [big[i] for i, _ in hits_b]).split())
    print(f"\nprompt words for Q1, k=3:  small chunks {words_s}"
          f"  big chunks {words_b}"
          f"  (whole document: {len(DOCUMENT.split())})")
    print("\nREAD THIS: same corpus, same questions, same maths. Giant chunks:")
    print(f"the gap on Q2 goes from {gaps[('small', 2)]:.3f} to"
          f" {gaps[('big', 2)]:.3f} -- a coin toss -- and {' and '.join(missed)}")
    print("both land on a chunk that does NOT contain the answer. The README")
    print("paragraph has all of Q3's words, twice. Q1 still hits, as sentence 5")
    print("of 7. And k=3 of 3 chunks is the whole document: the retriever did")
    print("nothing, you pasted the wiki into the prompt.")


def demo_8_two_hop():
    line("DEMO 8: BREAK IT -- a two-hop question with k=1")
    chunks = chunk(DOCUMENT)
    idf = fit_idf(chunks)
    index = build_index(chunks, idf)
    q = "Which port does the Lantern dashboard listen on?"
    print(f"Q: {q}")
    print("   needs chunk 8 (dashboard -> Beacon) AND chunk 9 (Beacon -> 8080)\n")
    for k in (1, 3):
        hits = retrieve(q, index, idf, k=k)
        print(f"k={k}:")
        show_hits(hits, chunks)
        got = {i for i, _ in hits}
        print(f"     hop 1 in prompt: {8 in got}   hop 2 in prompt: {9 in got}")
    print("\nsecond retrieval, query rewritten with hop 1's answer:")
    q2 = "Which port does Beacon listen on?"
    print(f"Q: {q2}")
    show_hits(retrieve(q2, index, idf, k=1), chunks, answer=9)
    print("\nREAD THIS: chunk 9 shares one word with the question ('port') and")
    print("everything with the ANSWER to hop 1 ('Beacon'). Single-shot retrieval")
    print("cannot see that. k=1 hands the model half a fact; k=3 happens to")
    print("catch it here, with a wrong port (7020) sitting next to it. The real")
    print("fix is to retrieve again after reading -- what agents call a loop.")


if __name__ == "__main__":
    demo_1_corpus_and_chunks()
    demo_2_embedding()
    demo_3_retrieval()
    demo_4_the_prompt()
    demo_5_raw_dot_product()
    demo_6_answer_not_in_corpus()
    demo_7_chunk_size()
    demo_8_two_hop()

    line("THE WHOLE PAPER, COMPRESSED")
    print("""
  1. chunk      = cut the corpus into passages (100 words in the paper)
  2. embed      = one fixed-size vector per passage. DPR/BERT there, tf-idf here
  3. index      = stack the vectors. FAISS at 21M rows, np.stack at 21
  4. retrieve   = embed the question, dot it against every row, keep the top k
  5. generate   = give the model [passage; question]; it writes the answer
  6. RAG-Seq    = one passage per generator call, k calls, average the answers
  7. RAG-Token  = same, but re-pick the passage at every generated token
  8. training   = fine-tune the query encoder and generator TOGETHER on
                  (question, answer) pairs. Document encoder frozen, index
                  built once. No labels for which passage to fetch.

  What everyone ships is steps 1-5 with a frozen retriever and a frozen
  model, all k passages in one prompt. This script is that, minus the model.
""")
