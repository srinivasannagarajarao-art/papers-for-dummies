"""
How a serving engine actually works -- for programmers, not researchers.

PagedAttention (Kwon et al., 2023, arXiv 2309.06180) plus the scheduling ideas
around it: continuous batching (Orca, Yu et al., OSDI 2022), prefix caching and
RadixAttention (SGLang, arXiv 2312.07104), prefill/decode disaggregation
(DistServe, arXiv 2401.09670).

Run it:      python3 serving_from_scratch.py
Debug it:    breakpoint in BlockAllocator.allocate and watch the block table.

No torch. No GPU. Nothing here is a benchmark -- it is a simulation with
counters, the same way the flash-attention page counts memory traffic instead
of timing a kernel. Every number printed below is a count, not a measurement.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the model we are serving, and the workload hitting it.
#
# Llama-2-7B shape, because the kv-cache page already did this arithmetic:
#   bytes per token = 2 * layers * kv_heads * head_dim * dtype_bytes
# ---------------------------------------------------------------------------
LAYERS, KV_HEADS, HEAD_DIM, DTYPE_BYTES = 32, 32, 128, 2
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES   # 512 KiB
KV_BUDGET_BYTES = 40 * 1024**3          # 40 GiB left for cache on an 80GB card
MAX_MODEL_LEN = 2048                    # the context length we advertise
BLOCK_TOKENS = 16                       # PagedAttention's page size


class Request:
    """One user request. Arrives, prefills its prompt, then decodes."""

    def __init__(self, rid, arrival, prompt_len, output_len, max_new):
        self.rid = rid
        self.arrival = arrival
        self.prompt_len = prompt_len
        self.output_len = output_len      # how many tokens it WILL generate
        self.max_new = max_new            # the cap the caller SET. Unknowable
                                          # in advance which of the two matters
        self.generated = 0                # how many it has generated so far
        self.start = None                 # step it entered the batch
        self.finish = None                # step it left

    @property
    def length(self):
        return self.prompt_len + self.generated

    def done(self):
        return self.generated >= self.output_len


def make_workload(n, seed=0):
    """Toy traffic: short prompts, wildly uneven output lengths.

    Uneven output length is the whole problem. If every reply were the same
    length, static batching would be fine and none of this would exist.
    """
    rng = np.random.default_rng(seed)
    reqs = []
    for i in range(n):
        arrival = int(rng.integers(0, 40))
        prompt = int(rng.integers(32, 256))
        out = int(rng.choice([8, 16, 32, 64, 128, 256],
                             p=[0.30, 0.25, 0.20, 0.15, 0.07, 0.03]))
        cap = int(rng.choice([64, 128, 256, 512, 1024]))
        reqs.append(Request(i, arrival, prompt, out, max(out, cap)))
    return sorted(reqs, key=lambda r: (r.arrival, r.rid))


# ---------------------------------------------------------------------------
# STAGE 1 -- STATIC BATCHING. Run-to-completion scheduling.
#
# Fill B slots, run the batch until the LONGEST member finishes, then swap the
# whole batch out. Every other slot sits idle from the moment its own sequence
# is done. This is `for batch in batches: process(batch)` -- the loop you would
# write first, and the loop Orca's iteration-level scheduling replaced.
# ---------------------------------------------------------------------------
def simulate_static_batching(reqs, slots):
    queue = list(reqs)
    step = 0
    busy_slot_steps = 0      # slot-steps that generated a token
    total_slot_steps = 0     # slot-steps available
    tokens = 0
    while queue:
        batch = queue[:slots]
        queue = queue[slots:]
        step = max(step, max(r.arrival for r in batch))
        for r in batch:
            r.start = step
        longest = max(r.output_len for r in batch)
        for t in range(longest):
            for r in batch:
                if r.generated < r.output_len:
                    r.generated += 1
                    tokens += 1
                    busy_slot_steps += 1
                    if r.done():
                        r.finish = step + t + 1
            total_slot_steps += slots      # all B slots are held, busy or not
        step += longest
    return dict(steps=step, tokens=tokens, busy=busy_slot_steps,
                capacity=total_slot_steps, reqs=reqs)


# ---------------------------------------------------------------------------
# STAGE 2 -- CONTINUOUS BATCHING (Orca's iteration-level scheduling).
#
# The scheduler runs between every decode iteration, not between batches. A
# finished sequence leaves immediately and the next queued request takes its
# slot on the very next step. Preemptive scheduling instead of run-to-
# completion; the batch is a mutable set, not a fixed array.
# ---------------------------------------------------------------------------
def simulate_continuous_batching(reqs, slots):
    queue = list(reqs)
    running = []
    step = 0
    busy_slot_steps = 0
    total_slot_steps = 0
    tokens = 0
    while queue or running:
        # admit: refill any free slot from whoever has arrived
        while len(running) < slots and queue and queue[0].arrival <= step:
            r = queue.pop(0)
            r.start = step
            running.append(r)
        if not running:                    # idle gap, nobody has arrived yet
            step = queue[0].arrival
            continue
        for r in running:                  # one iteration = one token each
            r.generated += 1
            tokens += 1
            busy_slot_steps += 1
        total_slot_steps += slots
        step += 1
        for r in running:
            if r.done() and r.finish is None:
                r.finish = step
        running = [r for r in running if not r.done()]
    return dict(steps=step, tokens=tokens, busy=busy_slot_steps,
                capacity=total_slot_steps, reqs=reqs)


# ---------------------------------------------------------------------------
# STAGE 3 -- CONTIGUOUS ALLOCATION. The thing PagedAttention deletes.
#
# You do not know how long the reply will be, so you reserve max_model_len
# contiguous tokens per request up front. Two separate wastes:
#   internal   -- reserved but never written (the reply was short)
#   external   -- free bytes that exist, but not in one contiguous run
# ---------------------------------------------------------------------------
def contiguous_churn(reqs, budget_tokens, fixed_reserve=None):
    """First-fit allocator over one arena, with requests arriving and leaving.

    Reservation is per request and made UP FRONT, because the allocator cannot
    grow a slab in place -- something else is sitting right behind it. Either
    you reserve max_model_len for everyone (fixed_reserve), or you reserve the
    caller's own max_new_tokens, which varies and so leaves ragged holes.
    """
    holes = [(0, budget_tokens)]
    live = {}                    # rid -> (start, size, used_tokens)
    admitted = rejected = peak = 0
    frag = None                  # state at the first rejection
    events = []
    for r in reqs:
        events.append((r.arrival, 0, r.rid, r))
        events.append((r.arrival + r.output_len, 1, r.rid, r))
    for _, kind, rid, r in sorted(events, key=lambda e: (e[0], e[1], e[2])):
        if kind == 1:                                  # request finishes
            if rid in live:
                start, size, _ = live.pop(rid)
                holes = merge_holes(holes + [(start, size)])
            continue
        want = fixed_reserve if fixed_reserve else r.prompt_len + r.max_new
        placed = None
        for i, (start, size) in enumerate(holes):      # first fit
            if size >= want:
                placed = start
                holes[i] = (start + want, size - want)
                break
        if placed is None:
            rejected += 1
            if frag is None:
                free = sum(sz for _, sz in holes)
                frag = (want, free, max([sz for _, sz in holes] + [0]))
            continue
        live[rid] = (placed, want, r.prompt_len + r.output_len)
        admitted += 1
        peak = max(peak, len(live))
    return dict(admitted=admitted, rejected=rejected, peak=peak, frag=frag)


def merge_holes(holes):
    """Coalesce adjacent free runs. Even after this, the holes stay ragged."""
    out = []
    for start, size in sorted(holes):
        if out and out[-1][0] + out[-1][1] == start:
            out[-1] = (out[-1][0], out[-1][1] + size)
        else:
            out.append((start, size))
    return out


def paged_churn(reqs, n_blocks, block_tokens=BLOCK_TOKENS):
    """Same workload, same memory, but blocks allocated one at a time."""
    alloc = BlockAllocator(n_blocks, block_tokens=block_tokens)
    live = {}
    admitted = rejected = peak = 0
    held = written = 0
    events = []
    for r in reqs:
        events.append((r.arrival, 0, r.rid, r))
        events.append((r.arrival + r.output_len, 1, r.rid, r))
    for _, kind, rid, r in sorted(events, key=lambda e: (e[0], e[1], e[2])):
        if kind == 1:
            if rid in live:
                live.pop(rid).release()
            continue
        need = r.prompt_len + r.output_len             # grown block by block
        if (need + block_tokens - 1) // block_tokens > len(alloc.free):
            rejected += 1
            continue
        s = Sequence(f"r{rid}", alloc)
        s.append_tokens(need)
        live[rid] = s
        admitted += 1
        peak = max(peak, len(live))
        held += len(s.table) * block_tokens
        written += s.n_tokens
    return dict(admitted=admitted, rejected=rejected, peak=peak,
                held=held, written=written, waste=held - written)


# ---------------------------------------------------------------------------
# STAGE 4 -- THE BLOCK ALLOCATOR. PagedAttention, in about forty lines.
#
# KV cache lives in fixed-size physical blocks. Each sequence gets a BLOCK
# TABLE: logical block i -> physical block number. The blocks need not be
# contiguous and are allocated only when the sequence actually reaches them.
# Refcounts make a block shareable between sequences; copy-on-write handles
# the moment they diverge. This is virtual memory, with the paper's own names.
# ---------------------------------------------------------------------------
class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, n_blocks, block_tokens=BLOCK_TOKENS):
        self.block_tokens = block_tokens
        self.free = list(range(n_blocks))      # a free list, nothing more
        self.refcount = {}                     # physical block -> refs
        self.n_blocks = n_blocks
        self.copies = 0                        # copy-on-write events

    def allocate(self):
        if not self.free:
            raise OutOfBlocks("no free physical blocks")
        b = self.free.pop(0)
        self.refcount[b] = 1
        return b

    def share(self, b):
        """Another sequence now points at this block. Do NOT copy it."""
        self.refcount[b] += 1
        return b

    def free_block(self, b):
        """The reference-counting guard: only the LAST owner really frees."""
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)
            del self.refcount[b]
            return True                        # actually returned to the pool
        return False                           # still shared, keep it alive

    def copy_on_write(self, b):
        """Writing into a shared block: take a private copy first."""
        if self.refcount[b] == 1:
            return b                           # sole owner, write in place
        new = self.allocate()
        self.refcount[b] -= 1
        self.copies += 1
        return new

    def used(self):
        return self.n_blocks - len(self.free)


class Sequence:
    """A sequence is a block table plus a length. That is the whole object."""

    def __init__(self, name, allocator):
        self.name = name
        self.alloc = allocator
        self.table = []          # logical block index -> physical block number
        self.n_tokens = 0

    def append_tokens(self, n):
        """Grow by n tokens, allocating a new physical block only when needed."""
        for _ in range(n):
            slot = self.n_tokens % self.alloc.block_tokens
            if slot == 0:                                  # current block full
                self.table.append(self.alloc.allocate())
            else:                                          # writing inside one
                last = len(self.table) - 1
                self.table[last] = self.alloc.copy_on_write(self.table[last])
            self.n_tokens += 1

    def fork(self, name):
        """Parallel sampling: same prompt, new sequence, shared blocks."""
        child = Sequence(name, self.alloc)
        child.table = [self.alloc.share(b) for b in self.table]
        child.n_tokens = self.n_tokens
        return child

    def release(self):
        for b in self.table:
            self.alloc.free_block(b)
        self.table = []

    def internal_waste(self):
        """Tokens of block space held but not written: only the last block."""
        return len(self.table) * self.alloc.block_tokens - self.n_tokens


# ---------------------------------------------------------------------------
# STAGE 5 -- PREFIX CACHE. Blocks of a shared prompt prefix are content-
# addressed: hash the tokens in the block, and if a block with that hash is
# already resident, point at it instead of prefilling it again. SGLang's
# RadixAttention keeps these hashes in a radix tree so partial prefixes share
# too; here a dict is enough to count the saving.
# ---------------------------------------------------------------------------
class PrefixCache:
    def __init__(self, block_tokens=BLOCK_TOKENS):
        self.block_tokens = block_tokens
        self.blocks = {}          # hash of prefix -> physical block (pretend)
        self.hits = 0
        self.misses = 0

    def prefill(self, tokens):
        """Returns the number of tokens actually recomputed."""
        recomputed = 0
        for i in range(0, len(tokens) - self.block_tokens + 1,
                       self.block_tokens):
            key = hash(tuple(tokens[:i + self.block_tokens]))   # prefix hash
            if key in self.blocks:
                self.hits += 1
            else:
                self.blocks[key] = len(self.blocks)
                self.misses += 1
                recomputed += self.block_tokens
        tail = len(tokens) % self.block_tokens
        return recomputed + tail

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ---------------------------------------------------------------------------
# STAGE 6 -- prefill vs decode, reusing the kv-cache page's counting.
# MACs and bytes moved, per the same approximate model: 4*d*d projections,
# 2*d*n attention, 2*d*ff feed-forward, and the weights read once per pass.
# ---------------------------------------------------------------------------
D_MODEL, D_FF, N_PARAMS = 4096, 11008, 6_740_000_000


def macs_per_token(n_ctx):
    per_layer = 4 * D_MODEL * D_MODEL + 2 * D_MODEL * n_ctx + 2 * D_MODEL * D_FF
    return LAYERS * per_layer


# A card's two spec-sheet numbers. Nothing here is measured on one -- this is
# a roofline: a pass takes as long as the SLOWER of its arithmetic and its
# memory traffic. A100-80GB shape: 312 TFLOP/s bf16, 2.0 TB/s of HBM.
PEAK_MACS_PER_S = 312e12 / 2          # one FLOP-pair per MAC
PEAK_BYTES_PER_S = 2.0e12


def pass_cost(n_tokens, n_ctx, batch=1):
    """MACs, bytes moved, and the roofline time in ms for one forward pass."""
    macs = n_tokens * batch * macs_per_token(n_ctx)
    bytes_moved = N_PARAMS * DTYPE_BYTES + n_ctx * batch * BYTES_PER_TOKEN
    ms = max(macs / PEAK_MACS_PER_S, bytes_moved / PEAK_BYTES_PER_S) * 1000
    return macs, bytes_moved, ms


def intensity(n_tokens, n_ctx, batch=1):
    """MACs per byte of memory traffic. High = compute-bound."""
    macs, bytes_moved, _ = pass_cost(n_tokens, n_ctx, batch)
    return macs / bytes_moved


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def demo_1_static_vs_continuous():
    """The opening argument: run-to-completion wastes most of the GPU."""
    line("DEMO 1: static batching vs continuous batching")

    slots = 8
    static = simulate_static_batching(make_workload(64, seed=1), slots)
    cont = simulate_continuous_batching(make_workload(64, seed=1), slots)

    for name, s in (("static", static), ("continuous", cont)):
        occ = s["busy"] / s["capacity"]
        print(f"\n{name} batching, {slots} slots, 64 requests")
        print(f"  iterations to drain the queue : {s['steps']:6d}")
        print(f"  tokens generated              : {s['tokens']:6d}")
        print(f"  slot-steps held               : {s['capacity']:6d}")
        print(f"  slot-steps that did work      : {s['busy']:6d}")
        print(f"  occupancy                     : {occ:6.1%}")
        print(f"  idle fraction                 : {1 - occ:6.1%}")
        print(f"  throughput (tokens/iteration) : {s['tokens'] / s['steps']:6.2f}")

    print("\noccupancy over time, static batching (first 3 batches):")
    reqs = sorted(static["reqs"], key=lambda r: r.rid)
    for b in range(3):
        batch = reqs[b * slots:(b + 1) * slots]
        outs = sorted(r.output_len for r in batch)
        longest = outs[-1]
        alive_steps = sum(outs)
        print(f"  batch {b}: output lengths {outs}")
        print(f"           runs for {longest:3d} steps, "
              f"slots busy {alive_steps / (longest * slots):5.1%}")

    print(f"\nspeedup in iterations: "
          f"{static['steps'] / cont['steps']:.2f}x  (same tokens generated)")
    print("\nREAD THIS: nothing about the model changed. The same tokens came")
    print("out of the same weights. Static batching just held slots hostage")
    print("until the longest reply in the batch finished. Orca called the fix")
    print("iteration-level scheduling; everyone else calls it continuous")
    print("batching.")


def demo_2_fragmentation():
    """PagedAttention's motivation: reserving up front is ruinous."""
    line("DEMO 2: contiguous reservation vs paged allocation")

    budget_tokens = KV_BUDGET_BYTES // BYTES_PER_TOKEN
    reqs = make_workload(400, seed=2)
    print(f"KV budget       : {KV_BUDGET_BYTES / 1024**3:.0f} GiB")
    print(f"bytes per token : {BYTES_PER_TOKEN:,} "
          f"({BYTES_PER_TOKEN / 1024:.0f} KiB)")
    print(f"budget in tokens: {budget_tokens:,}")
    print(f"requests        : {len(reqs)}, arriving and finishing over time")

    a = contiguous_churn(reqs, budget_tokens, fixed_reserve=MAX_MODEL_LEN)
    print(f"\ncontiguous, reserve max_model_len ({MAX_MODEL_LEN}) for everyone:")
    print(f"  peak concurrent requests : {a['peak']:6d}")
    print(f"  admitted / rejected      : {a['admitted']:6d} / {a['rejected']}")
    want, free, hole = a['frag']
    print(f"  at the first rejection   : wanted {want} tokens, "
          f"{free:,} free, biggest hole {hole}")

    b = contiguous_churn(reqs, budget_tokens)
    print("\ncontiguous, reserve each caller's own max_new_tokens:")
    print(f"  peak concurrent requests : {b['peak']:6d}")
    print(f"  admitted / rejected      : {b['admitted']:6d} / {b['rejected']}")
    want, free, hole = b['frag']
    print(f"  at the first rejection   : wanted {want} tokens, "
          f"{free:,} free, biggest hole {hole}")
    print(f"  external fragmentation   : {free - hole:,} free tokens that "
          f"exist but not in one piece")

    c = paged_churn(reqs, budget_tokens // BLOCK_TOKENS)
    print(f"\npaged, block size {BLOCK_TOKENS}, allocated on demand:")
    print(f"  peak concurrent requests : {c['peak']:6d}")
    print(f"  admitted / rejected      : {c['admitted']:6d} / {c['rejected']}")
    print(f"  tokens held / written    : {c['held']:,} / {c['written']:,}")
    print(f"  internal waste           : {c['waste']:,} "
          f"({c['waste'] / c['held']:.1%}, and never more than 15 per sequence)")
    print(f"  external fragmentation   : 0 -- every free block fits every "
          f"sequence")
    print(f"\npeak concurrency, same 40 GiB:")
    print(f"  {a['peak']} (reserve max_model_len) -> {b['peak']} (reserve "
          f"max_new_tokens) -> {c['peak']} (paged)")
    print(f"  = {c['peak'] / a['peak']:.1f}x more concurrent users, "
          f"and nobody was turned away")

    print("\nblock size sweep, same workload and same memory:")
    print("  block | peak | rejected | internal waste | table entries at peak")
    for bs in (1, 4, 16, 64, 256, 1024):
        r = paged_churn(reqs, budget_tokens // bs, block_tokens=bs)
        entries = r['peak'] * (r['held'] / max(r['admitted'], 1)) / bs
        print(f"  {bs:5d} | {r['peak']:4d} | {r['rejected']:8d} | "
              f"{r['waste']:14,} | {entries:12.0f}")
    print("\nREAD THIS: block size is the one knob. Too big and you are back")
    print("to reserving space you never write -- at 1024 the waste is worse")
    print("than the data. Too small and the block table itself becomes the")
    print("thing you are scanning: at block size 1 the engine is tracking")
    print("hundreds of thousands of entries to save a few hundred tokens.")
    print("vLLM ships 16.")


def demo_3_block_table():
    """The page table, small enough to read."""
    line("DEMO 3: the block table, printed")

    alloc = BlockAllocator(12, block_tokens=4)
    alloc.free = [5, 2, 9, 0, 7, 3, 11, 1, 8, 4, 6, 10]   # a used-up free list
    s = Sequence("seq-A", alloc)
    s.append_tokens(10)
    print("sequence of 10 tokens, block size 4")
    print("  logical block -> physical block")
    for i, p in enumerate(s.table):
        print(f"    {i} -> {p}")
    print(f"  physical blocks: {s.table}  <- NOT contiguous, and that is fine")
    print(f"  tokens {s.n_tokens}, blocks {len(s.table)}, "
          f"internal waste {s.internal_waste()} tokens (last block only)")

    print("\ndecode 3 more tokens, one at a time:")
    for _ in range(3):
        before = len(s.table)
        s.append_tokens(1)
        grew = "appended a new physical block" if len(s.table) > before \
            else "wrote into the current block"
        print(f"  n_tokens={s.n_tokens:3d} table={s.table}  {grew}")
    print("\nREAD THIS: attention reads the sequence through this table, so the")
    print("cache does not have to be one slab. Growth costs one block, not a")
    print("reallocation and a copy of the whole cache.")


def demo_4_copy_on_write():
    """Parallel sampling shares the prompt until the samples diverge."""
    line("DEMO 4: copy-on-write sharing of a shared prompt")

    block = 16
    prompt_len, n_samples = 90, 4
    alloc = BlockAllocator(4096, block_tokens=block)
    parent = Sequence("prompt", alloc)
    parent.append_tokens(prompt_len)
    print(f"prompt {prompt_len} tokens = {len(parent.table)} blocks: "
          f"{parent.table}")
    print(f"blocks used after prompt: {alloc.used()}")

    kids = [parent.fork(f"sample-{i}") for i in range(n_samples)]
    print(f"\nforked {n_samples} samples from it (parallel sampling):")
    print(f"  blocks used: {alloc.used()}  (unchanged -- all four share)")
    print(f"  refcounts  : "
          f"{[alloc.refcount[b] for b in parent.table]}")
    naive = (n_samples + 1) * len(parent.table)
    print(f"  naive copy would need {naive} blocks; we hold {alloc.used()}")
    print(f"  saved: {naive - alloc.used()} blocks = "
          f"{(naive - alloc.used()) * block * BYTES_PER_TOKEN / 1024**2:.1f} MiB")

    print("\nnow each sample generates its first token (they diverge):")
    for k in kids:
        before = alloc.copies
        k.append_tokens(1)
        print(f"  {k.name}: table={k.table}  "
              f"copy-on-write triggered: {alloc.copies > before}")
    print(f"  copy-on-write events: {alloc.copies}")
    print(f"  blocks used now     : {alloc.used()}  "
          f"(one private block each, the rest still shared)")

    print("\nthe reference-counting guard:")
    kids[0].release()
    shared = parent.table[0]
    print(f"  released sample-0. block {shared} refcount now "
          f"{alloc.refcount[shared]}")
    print(f"  block {shared} back on the free list? "
          f"{shared in alloc.free}  <- still referenced, so no")
    parent.release()
    for k in kids[1:]:
        k.release()
    print(f"  released everyone. block {shared} on the free list? "
          f"{shared in alloc.free}")
    print("\nREAD THIS: free_block decrements and only returns the block when")
    print("the count hits zero. Drop that check and sample-0 finishing would")
    print("hand a live prompt block to the next request -- silent corruption,")
    print("the same bug as a double free.")


def demo_5_prefix_cache():
    """A shared system prompt is prefill work you only do once."""
    line("DEMO 5: prefix caching across requests")

    rng = np.random.default_rng(3)
    system = list(range(300))            # a 300-token system prompt
    for shared_frac in (1.0, 0.5, 0.0):
        cache = PrefixCache()
        total, recomputed = 0, 0
        for _ in range(50):
            if rng.random() < shared_frac:
                prompt = system + list(rng.integers(1000, 9999, 40))
            else:
                prompt = list(rng.integers(1000, 9999, 340))
            total += len(prompt)
            recomputed += cache.prefill(prompt)
        print(f"\nfraction of requests using the shared system prompt: "
              f"{shared_frac:.0%}")
        print(f"  prompt tokens seen        : {total:7,}")
        print(f"  prompt tokens prefilled   : {recomputed:7,}")
        print(f"  prefill work saved        : "
              f"{1 - recomputed / total:6.1%}")
        print(f"  block cache hit rate      : {cache.hit_rate():6.1%}")
    print("\nREAD THIS: the saving is exactly the shared prefix, nothing more.")
    print("Hashing whole prefixes in a dict only matches from token 0. SGLang's")
    print("RadixAttention puts those prefixes in a radix tree instead, so two")
    print("requests sharing the first 300 tokens and diverging after still")
    print("share the 18 blocks they have in common, and eviction is LRU on the")
    print("tree's leaves.")


def demo_6_prefill_vs_decode():
    """Two different jobs sharing one GPU, and what that costs."""
    line("DEMO 6: prefill is compute-bound, decode is memory-bound")

    batch, ctx, prefill_len = 32, 512, 2048
    print("arithmetic intensity (MACs per byte moved), Llama-2-7B shape:")
    print(f"  prefill, {prefill_len}-token prompt : "
          f"{intensity(prefill_len, prefill_len):8.1f} MACs/byte")
    print(f"  decode, one token per seq, batch {batch}: "
          f"{intensity(1, ctx, batch):8.2f} MACs/byte")
    print("  a card needs roughly 100-300 to keep its arithmetic units busy")

    pm, pb, p_ms = pass_cost(prefill_len, prefill_len)
    dm, db, d_ms = pass_cost(1, ctx, batch)
    print(f"\nroofline time (A100-shaped: 156 TMAC/s, 2.0 TB/s):")
    print(f"  prefill {prefill_len} tokens : {pm / 1e9:9.1f} GMAC, "
          f"{pb / 1e9:6.1f} GB -> {p_ms:7.2f} ms  (compute-bound)")
    print(f"  decode step, batch {batch}: {dm / 1e9:9.1f} GMAC, "
          f"{db / 1e9:6.1f} GB -> {d_ms:7.2f} ms  (memory-bound)")

    horizon = 40
    for mode in ("shared GPU  ", "disaggregated"):
        clock, stamps = 0.0, []
        for step in range(horizon):
            if mode.strip() == "shared GPU" and step == 20:
                clock += p_ms              # a long prefill lands mid-stream
            clock += d_ms
            stamps.append(clock)
        gaps = np.diff([0.0] + stamps)
        print(f"\n{mode}: {batch} sequences decoding, {horizon} steps")
        print(f"  total wall time           : {clock:8.1f} ms")
        print(f"  median inter-token gap    : {np.median(gaps):8.2f} ms")
        print(f"  p95 inter-token gap       : {np.percentile(gaps, 95):8.2f} ms")
        print(f"  worst inter-token gap     : {gaps.max():8.2f} ms")
    print(f"\nthe stall: one token took {p_ms + d_ms:.1f} ms instead of "
          f"{d_ms:.1f} -- {(p_ms + d_ms) / d_ms:.1f}x")
    print(f"all {batch} decoding users saw it, and none of them sent that "
          f"prompt.")

    print("\nchunked prefill: same prefill, sliced, one chunk per iteration")
    for chunk in (2048, 512, 256, 128):
        _, _, c_ms = pass_cost(chunk, prefill_len)
        n_chunks = prefill_len // chunk
        print(f"  chunk {chunk:5d} -> {n_chunks:2d} iterations, "
              f"worst gap {c_ms + d_ms:7.2f} ms")
    print("\nREAD THIS: the two phases want different hardware. Prefill wants")
    print("FLOPs, decode wants memory bandwidth, and on one GPU every decode")
    print("step waits behind whatever prefill arrived. Chunked prefill caps")
    print("the stall but still spends decode time on prefill work. DistServe")
    print("(arXiv 2401.09670) runs them on separate GPUs so a prefill cannot")
    print("delay a decode at all, and each side is sized for its own")
    print("bottleneck.")


def demo_7_deadlock_and_preemption():
    """Admit with no free blocks and no eviction policy -- the engine wedges."""
    line("DEMO 7: no free blocks, no eviction -- deadlock, then preemption")

    alloc = BlockAllocator(8, block_tokens=16)
    seqs = []
    for i in range(4):
        s = Sequence(f"seq-{i}", alloc)
        s.append_tokens(32)                    # 2 blocks each = 8 blocks
        seqs.append(s)
    print(f"4 sequences x 2 blocks = {alloc.used()}/{alloc.n_blocks} blocks used")
    print(f"free blocks: {alloc.free}")

    print("\nevery sequence now wants one more token:")
    try:
        seqs[0].append_tokens(1)
    except OutOfBlocks as e:
        print(f"  seq-0 append -> OutOfBlocks: {e}")
        print("  no free block, no policy: the scheduler cannot advance ANY")
        print("  sequence, and none of them can finish without advancing.")
        print("  <- this is the deadlock")

    print("\nwith preemption (vLLM's swap-or-recompute, all-or-nothing):")
    victim = seqs[-1]
    freed = len(victim.table)
    victim.release()
    print(f"  preempt {victim.name}, evict its {freed} blocks, "
          f"push it back on the queue")
    print(f"  free blocks now: {alloc.free}")
    seqs[0].append_tokens(1)
    print(f"  seq-0 append -> ok, table={seqs[0].table}, "
          f"n_tokens={seqs[0].n_tokens}")
    print("\nREAD THIS: a preempted sequence loses its cache, not its tokens.")
    print("It is re-admitted later and its prompt is prefilled again (or its")
    print("blocks swapped back from host memory). Preemption is all-or-nothing")
    print("per sequence, because a half-evicted sequence cannot run at all.")


if __name__ == "__main__":
    demo_1_static_vs_continuous()
    demo_2_fragmentation()
    demo_3_block_table()
    demo_4_copy_on_write()
    demo_5_prefix_cache()
    demo_6_prefill_vs_decode()
    demo_7_deadlock_and_preemption()

    line("THE WHOLE THING, COMPRESSED")
    print("""
  1. static batching  = run-to-completion. Slots idle until the longest reply
                        in the batch finishes.
  2. continuous batch = schedule between iterations (Orca). A finished
                        sequence leaves; the queue refills its slot at once.
  3. contiguous KV    = reserve max_model_len per request. Internal waste plus
                        external fragmentation, and very few concurrent users.
  4. PagedAttention   = fixed-size blocks + a block table per sequence.
                        Virtual memory, applied to the KV cache.
  5. sharing          = refcount the blocks, copy on the first write.
  6. prefix caching   = a shared prompt is prefilled once (RadixAttention
                        keeps the prefixes in a tree).
  7. prefill/decode   = different bottlenecks; disaggregation gives them
                        different hardware.

  None of this touches attention itself. It is all storage layout and
  scheduling -- which is exactly why an engineer can read it.
""")
