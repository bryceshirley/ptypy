#!/bin/bash

# ==========================================
# Parameter Sweep Configuration
# ==========================================
# Define the values you want to test for blur and kz
OBJ_ZBLUR_VALUES=(0.005 0.05 0.5)
KZ_VALUES=(0.05 0.05 0.1)

# Fixed baseline values for other regularizers during this sweep
FIXED_ZTV=0.0
FIXED_L1=0.0
FIXED_TV=0.0

# Define static parameters for the python script
SCAN_NR=434
CROP=512
THICKNESS_UM=1580

# Define available CPU cores based on your htop screenshot.
# I am skipping Core 6 since it is currently at 98% utilization.
AVAILABLE_CORES=(0 1 2 3 4 5 7 8 9 10 11 12 13 14 15)
CORE_IDX=0

echo "Starting Multi-Core Ptychography Sweep..."

# ==========================================
# Execution Loop
# ==========================================
for obj_zblur in "${OBJ_ZBLUR_VALUES[@]}"; do
    for kz in "${KZ_VALUES[@]}"; do
        
        # Get the next available CPU core
        CORE_ID=${AVAILABLE_CORES[$CORE_IDX]}
        
        echo "Launching B=$obj_zblur, KZ=$kz on CPU Core $CORE_ID"
        
        # Run the python script pinned to the specific core in the background (&)
        # We redirect the output to a log file so it doesn't clutter your terminal
        LOG_FILE="sweep_obj_zblur${obj_zblur}_kz${kz}.log"
        taskset -c $CORE_ID ~/miniconda3/envs/ptypy_full/bin/python /home/dnz75396/ptypy/data/macros/ptycho/rec_multi_slice.py $SCAN_NR $CROP $THICKNESS_UM $FIXED_ZTV $FIXED_L1 $FIXED_TV $obj_zblur $kz > $LOG_FILE 2>&1 &
        
        # Move to the next core
        CORE_IDX=$((CORE_IDX + 1))
        
        # If we run out of assigned cores, wait for the current batch to finish before continuing
        if [ $CORE_IDX -ge ${#AVAILABLE_CORES[@]} ]; then
            echo "All assigned cores are busy. Waiting for batch to finish..."
            wait
            CORE_IDX=0
        fi
        
    done
done

# Wait for all background jobs to finish before exiting
wait
echo "Sweep completely finished!"
