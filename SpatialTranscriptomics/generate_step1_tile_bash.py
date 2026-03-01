import pickle
import os
import numpy as np

def main():
    # --- UNIQUE FILENAMES & DIRECTORIES ---
    # Distinct naming to avoid collision with pixel-level runs
    out_dir = 'stability_tile_metadata'
    script_name = 'run_step1_tile_stability.sh'
    
    # Configuration
    batch_size = 50
    DATA_FILE = 'predicted_genes_for_sensx.p'
    PYTHON_EXEC = 'python3'
    WORKER_SCRIPT = 'step1_tile_stability_worker.py'

    # 1. Create Output Directory
    if not os.path.exists(out_dir):
        print(f"Creating unique tile metadata directory: {out_dir}")
        os.makedirs(out_dir)

    # 2. Check total spots from data file
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, 'rb') as f:
        info_dict = pickle.load(f)
    
    total_spots = len(info_dict['all_XX'])
    
    # 3. Write bash script containing tile-specific commands
    with open(script_name, 'w') as f:
        #f.write("#!/bin/bash\n\n")
        #f.write(f"# Tile-Level Stability Scan (All 9 Tiles)\n")
        #f.write(f"# Output Directory: {out_dir}\n\n")
        
        for start in range(0, total_spots, batch_size):
            end = min(start + batch_size, total_spots)
            
            # Construct command targeting the tile worker
            cmd = (
                f"{PYTHON_EXEC} {WORKER_SCRIPT} "
                f"--start_idx {start} "
                f"--end_idx {end} "
                f"--out_dir {out_dir} "
                f"--time_qoi"
            )
            f.write(cmd + "\n")

    # 4. Final Permissions
    os.chmod(script_name, 0o755)
    num_commands = int(np.ceil(total_spots / batch_size))
    
    print(f"Successfully generated {script_name}")
    print(f"Total Spots: {total_spots}")
    print(f"Total Worker Commands: {num_commands}")
    print(f"Worker Script target: {WORKER_SCRIPT}")
    print("-" * 30)

if __name__ == "__main__":
    main()
