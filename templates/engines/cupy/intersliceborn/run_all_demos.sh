#!/usr/bin/env bash
# Run every demo on the free GPUs. Each demo owns one folder under
# results/, and its logs are written inside that folder alongside its
# reconstructions and figures.
#
# Pin GPUs by UUID, never by index: the index ordering is not stable
# between nvidia-smi query modes. List them with
#   nvidia-smi --query-gpu=uuid,memory.used --format=csv
set -u
cd "$(dirname "$0")"

NV=/home/dnz75396/ptyrad/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib:$NV/cufft/lib:$NV/cublas/lib:$NV/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(cd ../../../.. && pwd)"
export MPLBACKEND=Agg
PY=/home/dnz75396/ptyrad/.venv/bin/python

GPU_A=GPU-5c1cb7cd-b1a6-01b3-ef6e-ba1d766f59cb
GPU_B=GPU-b0d04bcb-6a95-8ece-f114-ff510679a5a8

D1=results/demo_01
D2=results/demo_02
D3=results/demo_03
mkdir -p "$D1" "$D2" "$D3"

run () {           # run <logfile> <gpu-uuid> <script> [args...]
  local log="$1" gpu="$2"; shift 2
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$@" > "$log" 2>&1
  echo "---- $(basename "$1" .py): exit $?, $((SECONDS - t0)) s -> $log ----"
}

echo "################ demo_01 + demo_02 in parallel ################"
run "$D1/run.log" "$GPU_A" demo_01_nanomax_engines.py --numiter 20 --out "$D1" &
P1=$!
run "$D2/run.log" "$GPU_B" demo_02_centred_nanomax.py --numiter 30 --order 1 --out "$D2" &
P2=$!
wait $P1 $P2

echo "################ demo_03, both arms in parallel ################"
run "$D3/run_isb.log" "$GPU_A" demo_03_centred_vs_multislice.py \
    --arm isb --numiter 20 --out "$D3" &
P3=$!
run "$D3/run_threepie.log" "$GPU_B" demo_03_centred_vs_multislice.py \
    --arm threepie --numiter 20 --out "$D3" &
P4=$!
wait $P3 $P4

echo "################ figures ################"
run "$D3/run_compare.log"     "$GPU_A" demo_03_centred_vs_multislice.py --arm plot --out "$D3"
run "$D3/run_plot_slices.log" "$GPU_A" demo_03_plot_slices.py --out "$D3" --iters 20
run "$D3/run_error_curve.log" "$GPU_A" plot_error_curve.py "$D3"
# notebook-style slice figure for every reconstruction produced above
run "$D1/run_plotting.log"    "$GPU_A" plotting.py "$D1"
run "$D3/run_plotting.log"    "$GPU_A" plotting.py "$D3"

echo "################ ALL DONE ################"
find results -name '*.png' -printf '%8s  %p\n' 2>/dev/null | sort -k2
