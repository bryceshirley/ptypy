#!/bin/bash
# Cell-sim pipeline: wait for GPU, simulate, first 10-iter FastBorn recon
source /home/dnz75396/miniconda3/etc/profile.d/conda.sh
conda activate ptypy_full
export PYTHONPATH=/home/dnz75396/ptypy
cd /home/dnz75396/ptypy
# wait for the ptyrad job on the GPU to finish
while kill -0 1466985 2>/dev/null; do sleep 60; done
echo "[pipeline] GPU free at $(date), starting cell simulation"
python run_cellsim.py > cellsim_sim.log 2>&1
rc=$?
echo "[pipeline] sim exit $rc"
if [ $rc -ne 0 ]; then echo "[pipeline] SIM FAILED"; exit 1; fi
echo "[pipeline] starting FastBorn recon: 10 iters, N=20, truth probe frozen"
python run_fastborn_cell.py CELL-A 10 1 max 0.5 1 20 3 100 1 0 0.0 stxm 0 0 0 0 none 0 \
  > cellrec_A.log 2>&1
echo "[pipeline] recon exit $?"
