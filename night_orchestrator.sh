#!/bin/bash
source /home/dnz75396/miniconda3/etc/profile.d/conda.sh
conda activate ptypy_full
# 1. wait for the 3PIE cell test report
while ! grep -q "report:" /home/dnz75396/ptypy/cellrec_T.log 2>/dev/null; do
  # bail out if the process died without a report
  pgrep -f run_fastthreepie_cell >/dev/null || { sleep 10; grep -q "report:" /home/dnz75396/ptypy/cellrec_T.log && break || { echo "[night] 3PIE died w/o report"; break; }; }
  sleep 30
done
echo "[night] cell test done at $(date); starting tBL-WSe2 N-sweep"
export PYTHONPATH=/home/dnz75396/ptyrad/src
cd /home/dnz75396/ptyrad/demo
python scripts/nsweep_paper.py > output/paper_extra_tests/nsweep_driver.log 2>&1
echo "[night] sweep finished at $(date)"
