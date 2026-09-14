#!/bin/bash
source /home/dnz75396/miniconda3/etc/profile.d/conda.sh
conda activate ptypy_full
export PYTHONPATH=/home/dnz75396/ptypy
cd /home/dnz75396/ptypy
while pgrep -f "run_fastborn_cell.py CELL-A" >/dev/null; do sleep 20; done
echo "[p3] sim (defocus +1400um) starting $(date)"
python run_cellsim.py > cellsim_sim2.log 2>&1 || { echo "[p3] SIM FAILED"; exit 1; }
echo "[p3] sim done; recon CELL-B (N=10) starting"
python run_fastborn_cell.py CELL-B 10 1 max 0.5 1 10 3 100 1 0 0.0 stxm 0 0 0 -700 none 0 > cellrec_B.log 2>&1
echo "[p3] recon exit $?"
