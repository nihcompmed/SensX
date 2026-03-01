import pickle
import os
import numpy as np

def main():
    # --- CONFIGURATION ---
    # Unique directory for pixel-level sensitivity results
    #out_dir = 'pixel_sensitivity_results'
    out_dir = 'pixel_sensitivity_results_50walks_1'
    tasks_per_worker = 1  # Recommended: 1 task per worker due to 10k groups
    script_name = 'run_step2_pixel_sensitivity.sh'
    
    # Files and Executables
    TASK_FILE = 'pixel_sensitivity_tasks.p'
    PYTHON_EXEC = 'python3'
    WORKER_SCRIPT = 'step2_center_tile_sensitivity_worker.py'

    # 1. Create Output Directory
    if not os.path.exists(out_dir):
        print(f"Creating pixel results directory: {out_dir}")
        os.makedirs(out_dir)

    # 2. Check for the Task File
    if not os.path.exists(TASK_FILE):
        print(f"Error: {TASK_FILE} not found. Run aggregate_pixel_stability.py first.")
        return

    with open(TASK_FILE, 'rb') as f:
        all_tasks = pickle.load(f)
    
    total_tasks = len(all_tasks)
    
    # 3. Write bash script
    with open(script_name, 'w') as f:
        #f.write("#!/bin/bash\n\n")
        #f.write(f"# Pixel-Level Sensitivity Phase (Step 2)\n")
        #f.write(f"# Targets: Center Tile (Tile 0), 10,000 RGB groups\n\n")
        
        for start in range(0, total_tasks, tasks_per_worker):
            end = min(start + tasks_per_worker, total_tasks)
            
            # Construct command
            cmd = (
                f"{PYTHON_EXEC} {WORKER_SCRIPT} "
                f"--task_file {TASK_FILE} "
                f"--task_start {start} "
                f"--task_end {end} "
                f"--out_dir {out_dir} "
                f"--time_qoi"
            )
            f.write(cmd + "\n")

    # 4. Permissions and Summary
    os.chmod(script_name, 0o755)
    num_commands = int(np.ceil(total_tasks / tasks_per_worker))
    
    print(f"Successfully generated {script_name}")
    print(f"Total Pixel Tasks: {total_tasks}")
    print(f"Total Worker Commands: {num_commands}")
    print(f"Results will be saved to: {out_dir}")

if __name__ == "__main__":
    main()
