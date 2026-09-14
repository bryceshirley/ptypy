import h5py
import matplotlib.pyplot as plt
import numpy as np

filename = '/home/dnz75396/ptypy/data/process/0002_multislice/scan_000434/ptycho_ptypy_crop_512_detdist_4.18_slice_0.00158/dumps/dump_scan_000434_FirstBornPIE_0010.ptyr'

with h5py.File(filename, 'r') as f:
    print("--- DATASETS IN DUMP FILE ---")
    f.visititems(lambda name, obj: print(name) if isinstance(obj, h5py.Dataset) else None)

with h5py.File(filename, 'r') as f:
    # Look at the exact structure if you ever get lost
    # f.visititems(lambda name, obj: print(name))
    
    # Extract the combined object (all slices multiplied)
    combined_obj = f['content/obj/Sscan00G00/data'][0]
    
    # Extract the incident probe
    probe = f['content/probe/Sscan00G00/data'][0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Phase of the combined object
    ax1.imshow(np.angle(combined_obj), cmap='gray')
    ax1.set_title("Combined Object Phase (Iteration 10)")
    
    # Plot Intensity of the probe
    ax2.imshow(np.abs(probe)**2, cmap='viridis')
    ax2.set_title("Probe Intensity (Iteration 10)")
    
    plt.tight_layout()
    plt.savefig('iteration_130_preview.png')
    print("Saved preview to iteration_130_preview.png")