- [ ] PRIORITY /home/dnz75396/butterfly-ref/interslice_born_demos/demo_out/sparse_chord_vs_tangent_recon.png still doesn't match ground truth. this must match may more iterations?
- [ ] Once done make sure /home/dnz75396/butterfly-ref/interslice_born_demos/theepiesimscan_torch.py and
/home/dnz75396/butterfly-ref/interslice_born_demos/issdiagnostics_torch.py aren't needed and the scripts
reproduce the results
- [x] order the demos.
- [x] add one to benchmark torch vs cupy implementations (demo_06: torch-cuda 1.2-2x faster than eager cupy per update; cupy closes the gap via engine-level cuda_graphs)
- [ ]add one to benchmark cupy interslice born series at a range of M as N increases
in terms of time per iteration against threepie_cupy. like /home/dnz75396/papers/figs-p1/cost_ptyrad_1x1_mutliple_scattering.py.
- [x] reproduce this /home/dnz75396/papers/figs-p1/paper1_M_vs_N.py study but with isb_cupy (demo_08; abtem unavailable -> documented synthetic volume; M* flat in N, set by contrast)
- [x] test on a thick weakly scattering (ie first or second born) sample with data generated from multislice (demo_09: eps_1 3.8e-2, orders 1/2/4 agree in data fit and projected phase)
- [ ] nanomax demostration compare threepie_cupy to isb_cupy

GROUND TRUTHS AND RECONSTRUCTIONS NEEDED FOR DEMOS
FIGURES MUST HAVE THE SAME PREFIX AS DEMO IE demo_01_*.png