#!/bin/bash
source /home/dnz75396/miniconda3/etc/profile.d/conda.sh
conda activate ptypy_full
export PYTHONPATH=/home/dnz75396/ptypy
cd /home/dnz75396/ptypy
python run_fastborn_cell.py CELL-3 10 1 max 0.5 1 3 3 100 1 0 0.0 stxm 0 0 0 -700 none 0 > cellrec_3B.log 2>&1
echo "[3B] exit $?"
python run_fastthreepie_cell.py T-CELL-3 10 1 max 0.5 1 3 3 100 1 0 0.0 exact stxm -700 > cellrec_3T.log 2>&1
echo "[3T] exit $?"
echo "CELL3 DONE"
