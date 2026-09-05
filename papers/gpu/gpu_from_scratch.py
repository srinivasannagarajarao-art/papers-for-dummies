"""
The GPU, for people who only ever call .cuda()

Run it:      python3 gpu_from_scratch.py
Debug it:    put a breakpoint in roofline() and step through one operation.

NumPy only. No GPU required, and nothing here is measured -- every hardware
number is a PUBLISHED SPECIFICATION from a vendor datasheet, and every result
is arithmetic against those specifications. Peak numbers are ceilings nobody
reaches. This script tells you which side of the roofline you are on, not how
fast your kernel will actually run.

The model is Williams, Waterman & Patterson, "Roofline: An Insightful Visual
Performance Model for Multicore Architectures", Communications of the ACM,
2009. It predates every accelerator named below and still explains all of them.
"""

import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# STAGE 0 -- the specifications.
#
# All figures are approximate, rounded, and taken from public vendor
# documentation (NVIDIA H100 Tensor Core GPU Architecture whitepaper and the
# H100 datasheet; NVLink and InfiniBand product pages). SXM form factor.
# They are PEAK numbers: the ceiling, not an achieved measurement.
#
# The latency column is the one exception and is flagged as such -- vendors do
# not publish memory latency, so those are order-of-magnitude figures from the
# published literature, good to a factor of about two and no better.
# ---------------------------------------------------------------------------
H100 = {
    "name":        "NVIDIA H100 SXM (published specification, approximate)",
    "sm_count":    132,              # streaming multiprocessors on the die
    "hbm_bytes":   80 * 1024**3,     # 80 GB of HBM3
    "hbm_bw":      3.35e12,          # ~3.35 TB/s memory bandwidth
    "l2_bytes":    50 * 1024**2,     # 50 MB L2, shared by all SMs
    "smem_bytes":  228 * 1024,       # up to 228 KB shared memory per SM
    "regs_bytes":  256 * 1024,       # 256 KB register file per SM
    # peak dense throughput, FLOP/s, no sparsity
    "fp32_novec":  67e12,            # plain FP32 CUDA cores, no tensor cores
    "tf32_tensor": 495e12,           # TF32 on tensor cores
    "fp16_tensor": 990e12,           # FP16/BF16 on tensor cores
    "fp8_tensor":  1979e12,          # FP8 on tensor cores
    "nvlink_bw":   900e9,            # ~900 GB/s NVLink per GPU, in a node
    "ib_bw":       50e9,             # 400 Gb/s InfiniBand NDR = ~50 GB/s
    "pcie_bw":     64e9,             # PCIe Gen5 x16, ~64 GB/s host<->device
}

# Two older parts, so we can watch the ridge point move over time.
# Same source class: published datasheets. Dense FP16 tensor throughput.
GENERATIONS = [
    # name,     HBM bandwidth B/s, dense FP16 tensor FLOP/s
    ("V100 SXM2 (2017)",  900e9,  125e12),
    ("A100 SXM  (2020)", 2039e9,  312e12),
    ("H100 SXM  (2022)", 3350e9,  990e12),
]


# ---------------------------------------------------------------------------
# STAGE 1 -- arithmetic intensity. The whole page is this one ratio.
#
#     intensity = FLOPs performed / bytes moved from memory
#
# Units: FLOP per byte. It is a property of the OPERATION, not of the chip.
# A matrix multiply has high intensity because every byte you load gets reused
# many times. An element-wise add has intensity near zero because every byte
# is touched once and thrown away.
# ---------------------------------------------------------------------------
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved


# ---------------------------------------------------------------------------
# STAGE 2 -- THE ROOFLINE. Williams, Waterman & Patterson, 2009.
#
#     attainable FLOP/s = min( peak FLOP/s,  bandwidth * arithmetic intensity )
#
# Two straight lines on a log-log plot. The slanted one is memory: if you can
# only move B bytes per second and you do I flops per byte, you can do at most
# B*I flops per second, no matter how good your arithmetic units are. The flat
# one is compute: you cannot exceed the chip's peak. The min of the two is the
# roof. Whichever term wins tells you what is limiting you.
# ---------------------------------------------------------------------------
def roofline(intensity, peak_flops, bandwidth):
    memory_bound_ceiling = bandwidth * intensity
    attainable = min(peak_flops, memory_bound_ceiling)
    limiter = "compute" if memory_bound_ceiling >= peak_flops else "memory"
    return attainable, limiter


# ---------------------------------------------------------------------------
# STAGE 3 -- the ridge point. Where the slanted roof meets the flat one.
#
#     ridge = peak FLOP/s / bandwidth      (units: FLOP per byte)
#
# Below this intensity you are memory-bound. Above it you are compute-bound.
# It is one number and it decides the fate of every kernel you will ever write.
# ---------------------------------------------------------------------------
def ridge_point(peak_flops, bandwidth):
    return peak_flops / bandwidth


# ---------------------------------------------------------------------------
# STAGE 4 -- FLOP and byte counts for the operations we care about.
#
# Each returns (name, flops, bytes). Byte counts assume the ideal case: each
# input read once from HBM, each output written once, everything else kept on
# chip. Real kernels move more. This is a lower bound on traffic, so the
# intensities below are an OPTIMISTIC estimate.
# ---------------------------------------------------------------------------
def op_elementwise_add(n, dtype_bytes=2):
    """c = a + b. Read two arrays, write one. One flop per element."""
    return ("element-wise add, n=%d" % n, n, 3 * n * dtype_bytes)


def op_matvec(n, dtype_bytes=2):
    """y = A @ x. The matrix is read once and each element used once."""
    flops = 2 * n * n
    bytes_moved = (n * n + n + n) * dtype_bytes
    return ("mat-vec %dx%d" % (n, n), flops, bytes_moved)


def op_matmul(n, dtype_bytes=2):
    """C = A @ B, all n x n. Three matrices in and out, n^3 work."""
    flops = 2 * n ** 3
    bytes_moved = 3 * n * n * dtype_bytes
    return ("mat-mul %dx%dx%d" % (n, n, n), flops, bytes_moved)


def op_attention(seq, d_head, dtype_bytes=2):
    """One attention head, fused: Q,K,V in, O out, scores never hit HBM.

    FLOPs: QK^T is 2*s*s*d, weights@V is another 2*s*s*d.
    Bytes: three inputs and one output of shape (s, d). This is the
    FlashAttention accounting -- the s x s score matrix stays on chip.
    """
    flops = 4 * seq * seq * d_head
    bytes_moved = 4 * seq * d_head * dtype_bytes
    return ("attention, seq=%d, d_head=%d" % (seq, d_head), flops, bytes_moved)


def op_decode_step(params, batch, dtype_bytes=2):
    """One token of autoregressive decode for a dense model.

    Every weight is read from HBM and used for 2 flops per token in the batch.
    That is the entire story of the KV-cache page, in two lines of arithmetic.
    """
    flops = 2 * params * batch
    bytes_moved = params * dtype_bytes
    return ("decode step, %.0fB params, batch=%d" % (params / 1e9, batch),
            flops, bytes_moved)


# ---------------------------------------------------------------------------
# STAGE 5 -- occupancy, crudely. How much of the machine does the work fill?
#
# An SM (streaming multiprocessor) is one of 132 independent processors on the
# die. A "warp" is a group of 32 threads that execute in lockstep; a kernel is
# chopped into thread blocks, and each block lands on exactly one SM. If your
# problem produces fewer blocks than there are SMs, the leftover SMs idle.
#
# We model the crudest version: n_blocks of equal cost, dealt round-robin.
# The kernel finishes when the busiest SM finishes.
# ---------------------------------------------------------------------------
def utilisation(n_blocks, n_sms):
    if n_blocks == 0:
        return 0.0
    waves = int(np.ceil(n_blocks / n_sms))     # how many full passes needed
    capacity = waves * n_sms                    # block-slots we paid for
    return n_blocks / capacity                  # fraction actually used


# ---------------------------------------------------------------------------
# STAGE 6 -- ring all-reduce cost. The gradient sync every data-parallel step.
#
# A ring all-reduce moves 2*(N-1)/N * S bytes through each link, where S is the
# size of the buffer and N the number of participants. The 2 is reduce-scatter
# then all-gather. This is bandwidth-optimal and is what NCCL actually does.
# ---------------------------------------------------------------------------
def ring_allreduce_bytes(buffer_bytes, n_ranks):
    if n_ranks < 2:
        return 0.0
    return 2.0 * (n_ranks - 1) / n_ranks * buffer_bytes


def transfer_time(bytes_moved, bandwidth):
    return bytes_moved / bandwidth


# ===========================================================================
# DEMOS
# ===========================================================================
def line(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def demo_1_the_hierarchy():
    """The supply chain: bench, shelf, warehouse, other building."""
    line("DEMO 1: the memory hierarchy is a supply chain")

    g = H100
    print("Accelerator: %s" % g["name"])
    print("All figures PUBLISHED SPECIFICATIONS, not measurements.\n")

    # (capacity, bandwidth, approximate latency in ns, metaphor)
    # Latency is order-of-magnitude from the literature, NOT a vendor spec.
    levels = [
        ("registers",     g["regs_bytes"],  "~100 TB/s",  1,    "on the bench"),
        ("shared mem/L1", g["smem_bytes"],  "~20 TB/s",   30,   "shelf behind you"),
        ("L2 cache",      g["l2_bytes"],    "~10 TB/s",   200,  "stockroom"),
        ("HBM3",          g["hbm_bytes"],   "3.35 TB/s",  500,  "warehouse in the yard"),
        ("NVLink peer",   0,                "900 GB/s",   2000, "the next building"),
        ("InfiniBand",    0,                "50 GB/s",    5000, "another site"),
    ]

    peak = g["fp16_tensor"]
    print("%-14s %9s %10s %9s %-21s %13s" %
          ("level", "capacity", "bandwidth", "latency*", "supply chain", "FLOPs missed"))
    print("-" * 81)
    for name, cap, bw, lat_ns, metaphor in levels:
        cap_s = "-" if cap == 0 else _human_bytes(cap)
        missed = peak * (lat_ns * 1e-9)   # arithmetic you could have done instead
        print("%-14s %9s %10s %6d ns %-21s %13d" %
              (name, cap_s, bw, lat_ns, metaphor, int(missed)))
    print("\n* latency is an order-of-magnitude figure, NOT a vendor spec.")

    print("\nREAD THIS: the last column is the point. While one HBM request is in")
    print("flight, an H100 at peak FP16 could have finished about %d floating" %
          int(peak * 500e-9))
    print("point operations. Reaching across NVLink costs about %d. The chip is" %
          int(peak * 2000e-9))
    print("not slow at arithmetic. It is starving. Every level down is a trip,")
    print("and the trips are what you are paying for.")
    print("\nCapacity runs the other way: %s of shared memory per SM against" %
          _human_bytes(g["smem_bytes"]))
    print("%s of HBM. The fast store is tiny. That asymmetry is the whole" %
          _human_bytes(g["hbm_bytes"]))
    print("reason FlashAttention exists.")


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%g %s" % (round(n, 2), unit)
        n /= 1024.0


def demo_2_roofline():
    """Compute the roofline for real operations and print which side they land."""
    line("DEMO 2: the roofline, computed -- which of your ops are memory-bound?")

    g = H100
    peak = g["fp16_tensor"]
    bw = g["hbm_bw"]
    ridge = ridge_point(peak, bw)

    print("Ceiling  : %.0f TFLOP/s dense FP16 tensor core (published peak)" % (peak / 1e12))
    print("Bandwidth: %.2f TB/s HBM3 (published peak)" % (bw / 1e12))
    print("Ridge    : %.1f FLOP/byte -- below this you are memory-bound\n" % ridge)

    ops = [
        op_elementwise_add(64 * 1024 * 1024),
        op_matvec(4096),
        op_matmul(4096),
        op_attention(512, 64),
        op_attention(8192, 64),
        op_decode_step(7e9, batch=1),
    ]

    print("%-32s %10s %14s %10s" % ("operation", "FLOP/byte", "attainable", "bound by"))
    print("-" * 70)
    for name, flops, byts in ops:
        ai = arithmetic_intensity(flops, byts)
        attain, limiter = roofline(ai, peak, bw)
        print("%-32s %10.3f %11.1f TF/s %10s" %
              (name, ai, attain / 1e12, limiter))

    print("\nREAD THIS: nothing here was measured. Every row is FLOPs divided by")
    print("bytes, compared against two published numbers. And it already tells")
    print("you the shape of your whole system:")
    print("  - element-wise add reaches %.1f%% of the chip's arithmetic peak." %
          (roofline(arithmetic_intensity(*op_elementwise_add(64 * 1024 * 1024)[1:]),
                    peak, bw)[0] / peak * 100))
    print("    Fusing element-wise ops is not micro-optimisation. It is the")
    print("    only thing that can help them.")
    print("  - big mat-mul is the one operation that is genuinely compute-bound.")
    print("  - attention crosses over as the sequence grows: seq=512 is")
    print("    memory-bound, seq=8192 is compute-bound. Same kernel.")
    print("  - a single-token decode step sits at %.2f FLOP/byte." %
          arithmetic_intensity(*op_decode_step(7e9, 1)[1:]))
    print("    That is %.0fx below the ridge. Batch-1 decode is a memory test" % (ridge / arithmetic_intensity(*op_decode_step(7e9, 1)[1:])))
    print("    wearing a compute costume. The KV-cache page argues this from")
    print("    the software side; this is the same claim from the hardware side.")


def demo_3_ridge_point_moves():
    """The ridge point rises every generation. That is why we keep getting
    more memory-bound even as chips get faster."""
    line("DEMO 3: the ridge point rises -- the field is getting MORE memory-bound")

    print("Dense FP16 tensor-core peak and HBM bandwidth, published specs.\n")
    print("%-20s %12s %14s %14s" %
          ("part", "HBM", "FP16 peak", "ridge FLOP/byte"))
    print("-" * 62)
    prev = None
    for name, bw, peak in GENERATIONS:
        r = ridge_point(peak, bw)
        print("%-20s %9.2f TB/s %10.0f TF/s %14.0f" % (name, bw / 1e12, peak / 1e12, r))
        prev = r

    v_bw, v_peak = GENERATIONS[0][1], GENERATIONS[0][2]
    h_bw, h_peak = GENERATIONS[-1][1], GENERATIONS[-1][2]
    print("\nOver those three parts:")
    print("  arithmetic throughput grew %.1fx" % (h_peak / v_peak))
    print("  memory bandwidth  grew %.1fx" % (h_bw / v_bw))
    print("  so the ridge point grew %.1fx" %
          (ridge_point(h_peak, h_bw) / ridge_point(v_peak, v_bw)))

    print("\nREAD THIS: arithmetic got cheap faster than bytes did. Every")
    print("generation, the intensity you need to stay compute-bound goes UP.")
    print("An operation that was comfortably compute-bound on a V100 can be")
    print("memory-bound on an H100 without a single line of code changing.")
    print("This is why the interesting systems papers of the last five years")
    print("are all about moving less data rather than doing less arithmetic.")


def demo_4_precision_is_a_throughput_knob():
    """Lower precision buys arithmetic AND bytes. Two wins, one change."""
    line("DEMO 4: precision is a throughput knob, not just a memory trick")

    g = H100
    bw = g["hbm_bw"]
    n = 4096

    print("Same operation: C = A @ B, %dx%dx%d. Only the dtype changes.\n" % (n, n, n))
    print("%-22s %8s %11s %7s %10s %9s" %
          ("precision", "bytes/el", "peak", "ridge", "FLOP/byte", "bound by"))
    print("-" * 72)

    settings = [
        ("FP32 (no tensor core)", 4, g["fp32_novec"]),
        ("TF32 tensor core",      4, g["tf32_tensor"]),
        ("FP16 tensor core",      2, g["fp16_tensor"]),
        ("FP8 tensor core",       1, g["fp8_tensor"]),
    ]
    for label, dbytes, peak in settings:
        _, flops, byts = op_matmul(n, dtype_bytes=dbytes)
        ai = arithmetic_intensity(flops, byts)
        attain, limiter = roofline(ai, peak, bw)
        print("%-22s %8d %6.0f TF/s %7.0f %10.0f %9s" %
              (label, dbytes, peak / 1e12, ridge_point(peak, bw), ai, limiter))

    _, f32, b32 = op_matmul(n, dtype_bytes=4)
    _, f8, b8 = op_matmul(n, dtype_bytes=1)
    print("\nSame maths, %d GFLOPs either way." % (f32 / 1e9))
    print("Bytes moved: %.1f MB at FP32, %.1f MB at FP8 -- %.0fx less traffic." %
          (b32 / 1e6, b8 / 1e6, b32 / b8))
    print("Peak arithmetic: %.0f TF/s at plain FP32 against %.0f TF/s at FP8" %
          (g["fp32_novec"] / 1e12, g["fp8_tensor"] / 1e12))
    print("-- %.0fx more, from the published ceilings." % (g["fp8_tensor"] / g["fp32_novec"]))

    print("\nREAD THIS: dropping precision moves you on the roofline TWICE. The")
    print("bytes halve, so your arithmetic intensity doubles and you slide right")
    print("along the slanted roof. And the flat roof itself lifts, because the")
    print("tensor cores do more per clock at narrower widths. Quantisation is")
    print("usually sold as a memory saving. It is also, on this hardware, an")
    print("arithmetic saving of the same size. See the quantisation page for what")
    print("it costs you in accuracy -- this page only counts the throughput.")


def demo_5_occupancy():
    """A small kernel leaves most of the machine idle. This is why we batch."""
    line("DEMO 5: occupancy -- a small batch leaves most of the GPU idle")

    g = H100
    sms = g["sm_count"]
    print("H100 SXM has %d streaming multiprocessors (published spec)." % sms)
    print("An SM is one independent processor. A thread block runs on exactly")
    print("one SM. Fewer blocks than SMs means idle silicon.\n")

    print("Model: one block per sequence in the batch (the batch-1 decode case).")
    print("%10s %10s %8s %14s %14s" %
          ("batch", "blocks", "waves", "utilisation", "SMs idle"))
    print("-" * 60)
    for batch in (1, 8, 32, 64, 132, 133, 264, 512):
        blocks = batch
        u = utilisation(blocks, sms)
        waves = int(np.ceil(blocks / sms))
        idle = waves * sms - blocks
        print("%10d %10d %8d %13.1f%% %14d" % (batch, blocks, waves, u * 100, idle))

    print("\nREAD THIS: at batch 1 you are using %.2f%% of the machine and %d of" %
          (utilisation(1, sms) * 100, sms - 1))
    print("the %d SMs have nothing to do. You have not bought a slow GPU. You" % sms)
    print("have bought a fast one and handed it one job. Notice batch=133 too:")
    print("%.1f%% -- one block past a full wave costs you almost a whole second" %
          (utilisation(133, sms) * 100))
    print("wave for a single extra unit of work. Sizes that are not multiples")
    print("of the SM count leave a quantised tail of idle time.")
    print("\nThis is the entire argument for batching in a serving stack:")
    print("batch-1 decode is memory-bound (demo 2) AND leaves the machine idle")
    print("(here). Batching fixes both at once, and costs you latency.")


def demo_6_scaling_out():
    """Within a node it is NVLink. Across nodes it is InfiniBand, and that
    order-of-magnitude drop is what shapes distributed training."""
    line("DEMO 6: scaling out -- NVLink is the next building, InfiniBand is a lorry")

    g = H100
    params = 7e9
    grad_bytes = params * 2          # BF16 gradients
    gpus_per_node = 8

    print("Published link bandwidths (approximate, per GPU):")
    print("  NVLink (in a node)   : %.0f GB/s" % (g["nvlink_bw"] / 1e9))
    print("  InfiniBand NDR 400G  : %.0f GB/s" % (g["ib_bw"] / 1e9))
    print("  Ratio                : %.0fx\n" % (g["nvlink_bw"] / g["ib_bw"]))

    print("All-reduce of %.0fB BF16 gradients = %.1f GB buffer.\n" %
          (params / 1e9, grad_bytes / 1e9))

    print("%-32s %11s %9s" % ("configuration", "bytes/GPU", "time"))
    print("-" * 62)
    intra = ring_allreduce_bytes(grad_bytes, gpus_per_node)
    t_intra = transfer_time(intra, g["nvlink_bw"])
    print("%-32s %8.1f GB %8.1f ms" %
          ("8 GPUs, 1 node, over NVLink", intra / 1e9, t_intra * 1e3))

    for nodes in (2, 4, 16):
        # Hierarchical: reduce-scatter in node, all-reduce the shard across
        # nodes over IB, all-gather back in node. The IB leg carries 1/8th.
        shard = grad_bytes / gpus_per_node
        inter = ring_allreduce_bytes(shard, nodes)
        t_inter = transfer_time(inter, g["ib_bw"])
        total = t_intra + t_inter
        print("%-32s %8.2f GB %8.1f ms  (%.0f%% on IB)" %
              ("  + %d nodes, over InfiniBand" % nodes, inter / 1e9,
               total * 1e3, t_inter / total * 100))

    shard2 = ring_allreduce_bytes(grad_bytes / gpus_per_node, 2)
    print("\nREAD THIS: at 2 nodes the InfiniBand leg carries %.1f%% of the bytes" %
          (shard2 / intra * 100))
    print("the NVLink leg does, and still takes the majority of the time,")
    print("because the link is %.0fx slower. Inside a node the GPUs are" %
          (g["nvlink_bw"] / g["ib_bw"]))
    print("effectively one machine. Across nodes they are")
    print("posting parcels. That single ratio is why expert parallelism, all-to-")
    print("all placement and communication/computation overlap are hard problems")
    print("and not implementation details. The DeepSeek-V3 page is what one team")
    print("did about it; this is the constraint they were doing it about.")


def demo_7_break_it():
    """Five wrong beliefs, each priced in the same arithmetic."""
    line("DEMO 7: break it -- five things that feel true and are not")

    g = H100
    bw = g["hbm_bw"]
    peak = g["fp16_tensor"]

    # --- (1) a faster chip does not fix a memory-bound kernel ---------------
    print("\n(1) 'We'll buy the faster chip.'")
    name, flops, byts = op_decode_step(7e9, batch=1)
    ai = arithmetic_intensity(flops, byts)
    a_now, lim_now = roofline(ai, peak, bw)
    a_2x, lim_2x = roofline(ai, peak * 2, bw)          # double the arithmetic
    a_bw, lim_bw = roofline(ai, peak, bw * 2)          # double the bandwidth
    print("    %s, %.2f FLOP/byte" % (name, ai))
    print("    today                        : %8.2f TF/s  (%s-bound)" % (a_now / 1e12, lim_now))
    print("    2x the arithmetic peak       : %8.2f TF/s  (%s-bound)  <-- unchanged" %
          (a_2x / 1e12, lim_2x))
    print("    2x the memory bandwidth      : %8.2f TF/s  (%s-bound)  <-- doubled" %
          (a_bw / 1e12, lim_bw))
    print("    Doubling the arithmetic bought exactly %.0f extra TFLOP/s." %
          ((a_2x - a_now) / 1e12))

    # --- (2) batch of one --------------------------------------------------
    print("\n(2) 'Batch size one is fine, it's just one request.'")
    u = utilisation(1, g["sm_count"])
    print("    utilisation %.2f%%, %d of %d SMs idle." %
          (u * 100, g["sm_count"] - 1, g["sm_count"]))
    print("    You are renting %d processors and using 1." % g["sm_count"])

    # --- (3) a stray FP32 op kills the tensor-core path --------------------
    print("\n(3) 'We cast the model to FP16, so we get the tensor-core speedup.'")
    n = 4096
    _, f16flops, b16 = op_matmul(n, dtype_bytes=2)
    ai16 = arithmetic_intensity(f16flops, b16)
    a16, _ = roofline(ai16, g["fp16_tensor"], bw)
    _, f32flops, b32 = op_matmul(n, dtype_bytes=4)
    ai32 = arithmetic_intensity(f32flops, b32)
    a32, _ = roofline(ai32, g["fp32_novec"], bw)
    print("    FP16 tensor path : %6.0f FLOP/byte -> %7.1f TF/s ceiling" % (ai16, a16 / 1e12))
    print("    FP32 CUDA-core   : %6.0f FLOP/byte -> %7.1f TF/s ceiling" % (ai32, a32 / 1e12))
    print("    One op that forces the FP32 path costs %.1fx of the ceiling." %
          (a16 / a32))
    print("    The intensity halved AND the roof dropped. Both directions, at once.")

    # --- (4) host-to-device transfer inside the loop -----------------------
    print("\n(4) 'The .to(device) call is inside the loop. It's only a copy.'")
    tensor_bytes = 4096 * 4096 * 2
    iters = 1000
    added = tensor_bytes * iters
    t_pcie = transfer_time(added, g["pcie_bw"])
    t_hbm = transfer_time(added, bw)
    print("    one 4096x4096 FP16 tensor = %.1f MB" % (tensor_bytes / 1e6))
    print("    %d iterations             = %.1f GB over PCIe" % (iters, added / 1e9))
    print("    at %.0f GB/s PCIe Gen5    = %.2f s of pure copying" %
          (g["pcie_bw"] / 1e9, t_pcie))
    print("    the same bytes over HBM   = %.2f s -- PCIe is %.0fx further away" %
          (t_hbm, t_pcie / t_hbm))
    print("    Hoisting it out of the loop: %.1f MB, once. %.0fx less traffic." %
          (tensor_bytes / 1e6, iters))

    # --- (5) splitting across nodes when it would fit in one ---------------
    print("\n(5) 'We'll shard it across two nodes for headroom.'")
    params = 7e9
    weights = params * 2
    print("    %.0fB params in BF16 = %.1f GB. One H100 has %s." %
          (params / 1e9, weights / 1e9, _human_bytes(g["hbm_bytes"])))
    print("    It fits. Sharding it across nodes anyway buys you, per step:")
    shard = weights / 8
    inter = ring_allreduce_bytes(shard, 2)
    print("    %.2f GB across InfiniBand at %.0f GB/s = %.1f ms of interconnect" %
          (inter / 1e9, g["ib_bw"] / 1e9, transfer_time(inter, g["ib_bw"]) * 1e3))
    print("    that did not previously exist, on every single step, forever.")

    print("\nREAD THIS: all five are the same mistake -- reasoning about the")
    print("arithmetic and forgetting the trips. Count bytes first. The FLOPs")
    print("are almost never what is stopping you.")


def demo_8_sanity_check():
    """Prove the roofline function is doing what the 2009 paper says, with a
    tiny numeric check rather than a claim."""
    line("DEMO 8: sanity check -- the roofline is genuinely just min() of two lines")

    peak, bw = H100["fp16_tensor"], H100["hbm_bw"]
    ridge = ridge_point(peak, bw)
    print("ridge = peak / bandwidth = %.4f FLOP/byte\n" % ridge)

    for ai in (ridge / 100, ridge / 2, ridge, ridge * 2, ridge * 100):
        attain, lim = roofline(ai, peak, bw)
        # below the ridge, attainable should equal bandwidth * intensity exactly
        predicted = bw * ai
        agrees = np.isclose(attain, min(peak, predicted))
        print("  intensity %10.2f -> %8.1f TF/s  %-7s  min() holds: %s" %
              (ai, attain / 1e12, lim, agrees))

    print("\nREAD THIS: no curve, no fitting, no measurement. Two straight lines")
    print("and a min(). Williams, Waterman and Patterson published this in 2009")
    print("for multicore CPUs and it survived the entire GPU era unchanged,")
    print("because it only assumes you have a compute ceiling and a bandwidth")
    print("ceiling. Every accelerator ever built has both.")


if __name__ == "__main__":
    demo_1_the_hierarchy()
    demo_2_roofline()
    demo_3_ridge_point_moves()
    demo_4_precision_is_a_throughput_knob()
    demo_5_occupancy()
    demo_6_scaling_out()
    demo_7_break_it()
    demo_8_sanity_check()

    line("THE WHOLE MODEL, COMPRESSED")
    print("""
  1. intensity  = FLOPs / bytes moved. A property of your OPERATION.
  2. ridge      = peak FLOP/s / bandwidth. A property of your CHIP.
  3. roofline   = min(peak, bandwidth * intensity). Williams et al., 2009.
  4. below the ridge you are memory-bound; a faster chip does nothing.
  5. the ridge rises every generation -- arithmetic outpaces bandwidth.
  6. lower precision moves you right AND lifts the roof. Two wins.
  7. too few thread blocks and the SMs idle no matter what the roofline says.
  8. NVLink is one building over. InfiniBand is another site entirely.

  Every hardware figure above is a PUBLISHED PEAK SPECIFICATION. Nothing here
  was measured, and no real kernel reaches a peak. The roofline tells you what
  is limiting you, not what you will get.
""")
