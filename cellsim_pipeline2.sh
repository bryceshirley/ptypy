#!/bin/bash
source /home/dnz75396/miniconda3/etc/profile.d/conda.sh
conda activate ptypy_full
export PYTHONPATH=/home/dnz75396/ptypy
cd /home/dnz75396/ptypy
echo "[pipeline2] sim starting $(date)"
python run_cellsim.py > cellsim_sim.log 2>&1 || { echo "[pipeline2] SIM FAILED"; exit 1; }
echo "[pipeline2] sim done, recon CELL-A starting"
python run_fastborn_cell.py CELL-A 10 1 max 0.5 1 20 3 100 1 0 0.0 stxm 0 0 0 0 none 0 > cellrec_A.log 2>&1
echo "[pipeline2] recon exit $?"
