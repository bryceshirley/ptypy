"""
Demo 6: forward + adjoint wall time of ONE per-position update -- the
inter-slice Born series versus sequential multislice -- on the CuPy
backends, as a function of the slice count N, eager AND CUDA-graph
compiled.

Reproduces the protocol of
~/papers/figs-p1/cost_ptyrad_1x1_mutliple_scattering.py (PtyRAD/torch)
with the production CuPy code of this repository:

- batch 1, 32 x 32 frames (the launch-latency regime, where the depth-M
  vs depth-N distinction is the whole cost), a single probe mode and a
  single object mode;
- slice sweep N in {1, 2, 4, 8, 16, 32, 64, 128};
- Born arm: ptypy.custom.interslice_born.isb_cupy.born_series_step at
  orders M = 1..5 -- one call is the full forward pass, detector
  projection, adjoint pass, object and probe gradients and both
  preconditioners;
- multislice arm: the sequential per-view update of
  ptypy.custom.threepie_cupy.ThreePIE_cupy, driven STANDALONE through its
  actual kernel objects -- AuxiliaryWaveKernel.build_aux2_no_ex,
  PaddedSlicePropagationKernel (nearfield PropagationKernel) fw/bw,
  ThreePIEWaveKernel.pr_to_aux/aux_to_pr, the far-field
  PropagationKernel, FourierUpdateKernel.fourier_deviation /
  fmag_update_nopbound, PoUpdateKernel.{pr,ob}_norm_local /
  ob_update_local / pr_update_local and MaxKernel -- in exactly the
  sequence of ThreePIE_cupy._update_view with every slice active, the
  probe update on and the error channels off (the engine's cheapest
  configuration);
- TWO execution modes per arm, the reference's two panels: eager
  (plain launches), and COMPILED -- the whole per-position update
  captured once into a CUDA graph and the replays timed, the CuPy
  equivalent of the reference's torch.compile(reduce-overhead / CUDA
  graphs). The capture follows the production recipe of
  ISBCupy._step/engine_initialize: a non-default stream
  (cp.cuda.Stream(non_blocking=True)); cp.cuda.Device().synchronize()
  before the stream first touches arrays prepared on the default stream
  (the engine_iterate ordering step); one eager warm-up call per arm
  with cp.cuda.set_allocator pointed at a PRIVATE cp.cuda.MemoryPool;
  begin_capture/end_capture under that same allocator; previous
  allocator restored afterwards. Captured kernels keep writing to their
  recorded scratch addresses on every replay, so the private pool keeps
  those blocks out of the shared pool, where later allocations could
  land inside them. The engine holds ONE pool for its two run-long
  graphs; this benchmark captures 48 short-lived graphs, so it holds
  one pool PER ARM, alive for the whole life of that arm's graph and
  released only after the graph is destroyed -- the same invariant
  (no recorded scratch block is ever reusable while its graph can still
  replay), adapted to many graphs. Capture is tried
  in the default mode first and retried in ThreePIE_cupy's own relaxed
  mode if the default refuses; an arm that still refuses would fall
  back to eager and be marked "capture_failed" in the json (none does
  on this machine -- both arms capture fully, nothing stays eager);
- warm-up, then many repetitions with cp.cuda.Device().synchronize()
  around every repetition (replays of the captured graph in compiled
  mode); the reference's statistic (median over reps) is reported and
  the min and interquartile range are stored too.

Deviations from the reference, by necessity of the backends:
- the units of work differ slightly in what follows the adjoint: the
  ThreePIE chain APPLIES its sequential ePIE object/probe updates (they
  are fused into its kernels), while born_series_step returns gradients
  and preconditioners and leaves the Jacobi step to the engine; both are
  the complete per-position device work of their engines.
- the multislice magnitudes are the model's own (taken from one forward
  pass), so the fmag factor is ~1 and the ePIE updates do not drift the
  arrays over hundreds of repetitions; the Born core never mutates its
  inputs.
- the figure is drawn log-log (one line per Born order M plus the
  multislice line, the reference's colours/markers/labels, two panels
  eager | compiled) so the crossovers are visible.
- born_series_step clamps M to N (the series is nilpotent and terminates
  after N orders), so at N < M the order-M point coincides with the
  order-N one; the torch reference evaluated all M orders regardless.

The compiled panel is the primary result; the crossover table is
recomputed from it (stored under "compiled" in the json next to the
eager sections).

Run (recomputes if demo_out/demo_06_backend_benchmark.json is
missing; if the json exists but has no "compiled" section, only the
compiled pass is run and merged in):
    python demo_06_backend_benchmark.py [--recompute] [--warmup 10]
                                        [--reps 60]
"""

import os
import sys

# the demos live one level below the repo root; run in place against it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import statistics
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_out")
BASE = "demo_06_backend_benchmark"
JSON = os.path.join(OUT, BASE + ".json")

SLICES = [1, 2, 4, 8, 16, 32, 64, 128]
ORDERS = [1, 2, 3, 4, 5]
NPIX = 32
DZ = 3e-4  # slice separation (m) in the optical-bench geometry below

ORDINAL = {1: "1$^{st}$", 2: "2$^{nd}$", 3: "3$^{rd}$", 4: "4$^{th}$",
           5: "5$^{th}$"}


def crossover(sect):
    """Per order M, the smallest N from which order-M Born stays faster
    than sequential multislice for every larger N too."""
    out = {}
    for M in ORDERS:
        cross = None
        for i, N in enumerate(SLICES):
            if all(sect[f"born_M{M}"][str(n)]["median_ms"]
                   < sect["multislice"][str(n)]["median_ms"]
                   for n in SLICES[i:]):
                cross = N
                break
        out[str(M)] = cross
    return out


# ───────────────────────────── compute ─────────────────────────────────────
def recompute(warmup, reps, do_eager, do_compiled, prev=None):
    import cupy as cp

    from ptypy import utils as u
    from ptypy.core import geometry
    from ptypy.core import Base as theBase
    from ptypy.accelerate.cuda_cupy.kernels import (
        AuxiliaryWaveKernel, FourierUpdateKernel, PoUpdateKernel,
        PropagationKernel, ThreePIEWaveKernel)
    from ptypy.accelerate.cuda_cupy.array_utils import MaxKernel
    from ptypy.custom.multislice_utils import PaddedSlicePropagationKernel
    from ptypy.custom.interslice_born.isb_cupy import born_series_step

    Base = type("Base", (theBase,), {})  # dictionary-access owner for Geo

    def make_geo(propagation, distance, psize):
        P = Base()
        P.CType = np.complex64
        P.Ftype = np.float32
        g = u.Param()
        g.energy = None
        g.lam = 5.32e-7
        g.distance = distance
        g.psize = psize
        g.shape = NPIX
        g.propagation = propagation
        return geometry.Geo(owner=P, pars=g)

    # One far-field geometry for both arms (the propagation-kernel test's
    # optical bench), and one nearfield geometry per inter-slice gap; all
    # gaps are DZ, so a single nearfield kernel serves every gap (the
    # engine builds one per gap, but they are identical objects and the
    # per-update cost is the same).
    geo = make_geo("farfield", 15e-2, 24e-6)
    res = float(np.mean(geo.resolution))
    geo_nf = make_geo("nearfield", DZ, res)

    lam = geo.lam
    rng = np.random.default_rng(0)

    # shared probe and detector-plane bookkeeping
    pr0 = (rng.standard_normal((NPIX, NPIX))
           + 1j * rng.standard_normal((NPIX, NPIX)))
    pr0 = (pr0 / np.sqrt((np.abs(pr0) ** 2).sum())).astype(np.complex64)

    # far-field pair for the Born arm through ptypy's accelerate
    # PropagationKernel, exactly as CupyISBEngineInfra._build_farfield
    # wires it into the engine (and the same FFT machinery the
    # multislice arm uses, so the comparison shares its backend)
    aux_ff = cp.zeros((1, NPIX, NPIX), dtype=cp.complex64)
    PROP_B = PropagationKernel(aux_ff, geo.propagator, None, "cuda")
    PROP_B.allocate()
    _ff_fw_buf = aux_ff
    _ff_bw_buf = cp.zeros_like(aux_ff)

    def ff_fw(w):
        v = w.reshape(-1, NPIX, NPIX)
        PROP_B.fw(v, _ff_fw_buf)
        return _ff_fw_buf.reshape(w.shape)

    def ff_bw(w):
        v = w.reshape(-1, NPIX, NPIX)
        PROP_B.bw(v, _ff_bw_buf)
        return _ff_bw_buf.reshape(w.shape)

    def make_object(N):
        amp = np.clip(1.0 + 0.05 * rng.standard_normal((N, NPIX, NPIX)),
                      0.1, None)
        phase = 0.3 * rng.standard_normal((N, NPIX, NPIX))
        return (amp * np.exp(1j * phase)).astype(np.complex64)

    def make_H(N):
        """Angular-spectrum stack H_{z_j}, j = 0..N-1, as the ISB engine
        builds it: H_fwd[0] = 1 (entrance plane)."""
        f = np.fft.fftfreq(NPIX)
        a2 = ((f[:, None] * lam / res) ** 2 + (f[None, :] * lam / res) ** 2)
        kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
        H = np.ones((N, 1, 1, NPIX, NPIX), dtype=np.complex128)
        for j in range(1, N):
            H[j, 0, 0] = np.exp(2j * np.pi * (j * DZ / lam) * kz) * (a2 <= 1)
        return cp.asarray(H, dtype=cp.complex64)

    def timeit(fn):
        for _ in range(warmup):
            fn()
        cp.cuda.Device().synchronize()
        ts = []
        for _ in range(reps):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            fn()
            cp.cuda.Device().synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return {"median_ms": statistics.median(ts), "min_ms": ts[0],
                "iqr_ms": [ts[len(ts) // 4], ts[3 * len(ts) // 4]]}

    # ----------------------------------------------- CUDA graph capture
    # the production recipe of ISBCupy._step: the engine stream is
    # non-default (capture is illegal on the legacy default stream) and
    # warm-up AND capture allocate from a private pool that lives for the
    # whole benchmark -- captured kernels keep writing to the recorded
    # scratch addresses on every replay, and blocks returned to the
    # shared pool would be handed to later allocations and silently
    # overwritten.
    stream = cp.cuda.Stream(non_blocking=True)

    def compile_graph(fn, graph_pool):
        """Warm up once eagerly, then capture fn into a CUDA graph; both
        inside the arm's private pool. Returns a replay closure, or None
        if the chain refuses capture even in ThreePIE_cupy's relaxed
        mode."""
        # ISBCupy.engine_iterate's ordering step: the arm was just built
        # on the default stream (mag / I_dat seeding); order that before
        # the engine stream touches the same arrays.
        cp.cuda.Device().synchronize()
        prev_alloc = cp.cuda.get_allocator()
        cp.cuda.set_allocator(graph_pool.malloc)
        try:
            with stream:
                fn()  # eager warm-up: cuFFT plans, kernel compiles, pool
            stream.synchronize()
            graph = None
            for mode in (None, cp.cuda.runtime.streamCaptureModeRelaxed):
                try:
                    with stream:
                        if mode is None:
                            stream.begin_capture()
                        else:
                            stream.begin_capture(mode=mode)
                        fn()
                        graph = stream.end_capture()
                    break
                except Exception as e:  # noqa: BLE001 - capture errors
                    try:
                        stream.end_capture()
                    except Exception:  # noqa: BLE001
                        pass
                    err = e
            if graph is None:
                print(f"capture failed ({err}); arm stays eager")
                return None
        finally:
            cp.cuda.set_allocator(prev_alloc)
        return lambda: graph.launch(stream)

    # ----------------------------------------------------- multislice arm
    def build_threepie(N, obj_np):
        """The full _update_view kernel chain of ThreePIE_cupy for one
        view, standalone, and a closure that runs it once."""
        aux = cp.zeros((1, NPIX, NPIX), dtype=cp.complex64)
        AWK = AuxiliaryWaveKernel(None)
        FUK = FourierUpdateKernel(aux, nmodes=1)
        FUK.allocate()
        POK = PoUpdateKernel(None)
        TWK = ThreePIEWaveKernel(None)
        MAXK = MaxKernel(None)
        PROP = PropagationKernel(aux, geo.propagator, None, "cuda")
        PROP.allocate()
        prop_nf = PropagationKernel(aux, geo_nf.propagator, None, "cuda")
        prop_nf.allocate()
        slice_PROP = [PaddedSlicePropagationKernel(prop_nf, (NPIX, NPIX),
                                                   pad=1)] * max(N - 1, 1)

        ob_layers = [cp.asarray(obj_np[s][None]) for s in range(N)]
        pr_layers = [cp.asarray(pr0[None]) for _ in range(N)]
        slice_exits = [cp.empty_like(aux) for _ in range(N)]
        slice_tmp = cp.empty_like(aux)
        slice_back = cp.empty_like(aux)
        ex = cp.zeros_like(aux)
        obn = cp.zeros((1, NPIX, NPIX), dtype=cp.float32)
        prn = cp.zeros((1, NPIX, NPIX), dtype=cp.float32)
        obn_max = [cp.zeros((1,), dtype=cp.float32) for _ in range(N)]
        prn_max = [cp.zeros((1,), dtype=cp.float32) for _ in range(N)]
        addr = cp.zeros((1, 1, 5, 3), dtype=cp.int32)
        ma = cp.ones((1, NPIX, NPIX), dtype=cp.float32)

        def forward():
            for s in range(N):
                AWK.build_aux2_no_ex(slice_exits[s], addr, ob_layers[s],
                                     pr_layers[s])
                if s < N - 1:
                    slice_PROP[s].fw(slice_exits[s], slice_tmp)
                    TWK.aux_to_pr(pr_layers[s + 1], slice_tmp, addr)
            cp.copyto(ex, slice_exits[-1][:1])
            AWK.make_aux(aux, addr, ob_layers[-1], pr_layers[-1], ex,
                         c_po=1.0, c_e=0.0)
            PROP.fw(aux, aux)

        # model magnitudes: fmag factor ~1, so the ePIE updates stay ~0
        # over the repetitions and nothing drifts
        forward()
        mag = cp.abs(aux)

        def update_view():
            # EPIE coefficients: _c=1, _a=0, _b=1, _ob_a=_pr_a=0,
            # _ob_b=_pr_b=1 (alpha=beta=1), object norm local
            forward()
            FUK.fourier_deviation(aux, addr, mag)
            FUK.fmag_update_nopbound(aux, addr, mag, ma)
            PROP.bw(aux, aux)
            AWK.make_exit(aux, addr, ob_layers[-1], pr_layers[-1], ex,
                          c_a=1.0, c_po=0.0, c_e=-1.0)
            back_wave = ex
            for s in range(N - 1, -1, -1):
                if s < N - 1:
                    TWK.pr_to_aux(slice_tmp, pr_layers[s + 1], addr)
                    slice_PROP[s].bw(slice_tmp, slice_back)
                    back_wave = slice_back
                POK.pr_norm_local(addr, pr_layers[s], prn)
                MAXK.max(prn, prn_max[s])
                POK.ob_update_local(addr, ob_layers[s], pr_layers[s],
                                    back_wave, slice_exits[s], prn,
                                    a=0.0, b=1.0, prn_max=prn_max[s])
                POK.ob_norm_local(addr, ob_layers[s], obn)
                MAXK.max(obn, obn_max[s])
                POK.pr_update_local(addr, pr_layers[s], ob_layers[s],
                                    back_wave, slice_exits[s], obn,
                                    obn_max[s], a=0.0, b=1.0)

        return update_view

    # --------------------------------------------------------- Born arm
    def build_born(N, obj_np, M):
        O = cp.asarray(obj_np[:, None])          # [N, B=1, Ny, Nx]
        P = cp.asarray(pr0[None])                # [Mp=1, Ny, Nx]
        H_fwd = make_H(N)
        H_rev = cp.conj(H_fwd)
        # measured data for the detector projection: the model's own
        # intensities from a first call (the factor is then ~1; the core
        # never mutates its inputs either way)
        I_dat = cp.ones((1, NPIX, NPIX), dtype=cp.float32)
        I_dat = born_series_step(O, P, I_dat, H_fwd, H_rev, ff_fw, ff_bw,
                                 M=M)[5].astype(cp.float32)

        def step():
            born_series_step(O, P, I_dat, H_fwd, H_rev, ff_fw, ff_bw,
                             M=M, coeffs=None, mask=None,
                             calc_probe_grad=True)

        return step

    out = prev if prev is not None else {}
    out.setdefault("config", {}).update(dict(
        batch=1, Npix=NPIX, pmode=1, omode=1, warmup=warmup, reps=reps,
        stat="median", orders=ORDERS, sweep="slices", dz_m=DZ,
        multislice="ThreePIE_cupy _update_view kernel chain, standalone",
        compiled="CUDA graph capture/replay, ISBCupy._step recipe "
                 "(non-default stream, private MemoryPool per arm, "
                 "alive for the graph's life)"))
    out["env"] = {"cupy": cp.__version__,
                  "gpu": cp.cuda.runtime.getDeviceProperties(0)[
                      "name"].decode()}
    out["slices"] = SLICES
    if do_eager:
        out["multislice"] = {}
        for M in ORDERS:
            out[f"born_M{M}"] = {}
    if do_compiled:
        out["compiled"] = {"multislice": {}}
        for M in ORDERS:
            out["compiled"][f"born_M{M}"] = {}

    def run_arm(key, label, fn):
        if do_eager:
            r = timeit(fn)
            out[key][str(N)] = r
            print(f"N={N:3d} {label} eager    {r['median_ms']:7.2f} ms "
                  f"(min {r['min_ms']:6.2f})", flush=True)
        if do_compiled:
            # one private pool per captured arm, alive for the whole life
            # of its graph: the recorded scratch addresses can then never
            # be handed to anything else while the graph can still replay
            # (ISBCupy holds one pool for its two run-long graphs; here
            # every arm's graph lives only for its timing)
            graph_pool = cp.cuda.MemoryPool()
            replay = compile_graph(fn, graph_pool)
            if replay is None:
                r = dict(timeit(fn), capture_failed=True)
            else:
                r = timeit(replay)
            out["compiled"][key][str(N)] = r
            print(f"N={N:3d} {label} compiled {r['median_ms']:7.2f} ms "
                  f"(min {r['min_ms']:6.2f})"
                  + (" CAPTURE FAILED, eager" if replay is None else ""),
                  flush=True)
            cp.cuda.Device().synchronize()
            del replay                       # graph first ...
            graph_pool.free_all_blocks()     # ... then its scratch
            del graph_pool

    for N in SLICES:
        obj_np = make_object(N)
        fn = build_threepie(N, obj_np)
        run_arm("multislice", "multislice", fn)
        for M in ORDERS:
            run_arm(f"born_M{M}", f"born M={M} ", build_born(N, obj_np, M))
        cp.get_default_memory_pool().free_all_blocks()

    if do_eager:
        out["crossover_N"] = crossover(out)
        print("eager crossovers:", out["crossover_N"])
    if do_compiled:
        out["compiled"]["crossover_N"] = crossover(out["compiled"])
        print("compiled crossovers:", out["compiled"]["crossover_N"])

    os.makedirs(OUT, exist_ok=True)
    with open(JSON, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", JSON)


# ─────────────────────────────── plot ──────────────────────────────────────
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
        "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.7,
        "axes.edgecolor": "#9CA3AF", "axes.linewidth": 0.9,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "white"})
    r = json.load(open(JSON))
    orders = r["config"].get("orders", ORDERS)
    series = [("multislice", "multislice (ThreePIE_cupy kernels)",
               "#2d7dd2", "o")] + [
        (f"born_M{M}", f"Inter-slice {ORDINAL[M]} Born", col, mk)
        for M, col, mk in zip(orders,
                              ["#f4661b", "#2aa876", "#5f3dc4",
                               "#e8719e", "#0e7490"],
                              ["s", "^", "D", "v", "P"])]
    N = r["slices"]
    panels = [(r, "eager")]
    if "compiled" in r:
        panels.append((r["compiled"], "CUDA graphs (captured replay)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.2),
                             squeeze=False)
    for ax, (data, title) in zip(axes[0], panels):
        for key, lab, col, mk in series:
            y = [data[key][str(n)]["median_ms"] for n in N]
            ax.loglog(N, y, "-" + mk, color=col, ms=8, lw=2.2, label=lab)
        ax.set_xticks(N, [str(n) for n in N])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("slices $N$")
        ax.set_ylabel("forward + adjoint (ms)")
        ax.set_title(title + ", batch 1, %dx%d" % (r["config"]["Npix"],
                                                   r["config"]["Npix"]),
                     fontsize=11)
    axes[0][0].legend(fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, BASE + ".png"), bbox_inches="tight")
    print("wrote", os.path.join(OUT, BASE + ".png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--recompute", action="store_true",
                    help="re-run both the eager and the compiled pass")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()
    prev = json.load(open(JSON)) if os.path.exists(JSON) else None
    if a.recompute or prev is None:
        recompute(a.warmup, a.reps, do_eager=True, do_compiled=True)
    elif "compiled" not in prev:
        # keep the stored eager data, add the compiled pass only
        recompute(a.warmup, a.reps, do_eager=False, do_compiled=True,
                  prev=prev)
    plot()
