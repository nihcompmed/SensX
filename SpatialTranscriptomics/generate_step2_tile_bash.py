import pickle
import os
import numpy as np

def main():
    # --- CONFIGURATION ---
    # Unique directory for tile-level sensitivity results
    out_dir = 'tile_sensitivity_results'
    
    # OPTIMIZATION: 
    # Since we are only computing 9 groups (vs 10,000), each task is very fast.
    # We increase tasks_per_worker to avoid flooding the job scheduler.
    tasks_per_worker = 50 
    
    script_name = 'run_step2_tile_sensitivity.sh'
    
    # Files and Executables
    TASK_FILE = 'tile_sensitivity_tasks.p'
    PYTHON_EXEC = 'python3'
    WORKER_SCRIPT = 'step2_tile_sensitivity_worker.py'

    # 1. Create Output Directory
    if not os.path.exists(out_dir):
        print(f"Creating tile results directory: {out_dir}")
        os.makedirs(out_dir)

    # 2. Check for the Task File
    if not os.path.exists(TASK_FILE):
        print(f"Error: {TASK_FILE} not found. Run aggregate_tile_stability.py first.")
        return

    with open(TASK_FILE, 'rb') as f:
        all_tasks = pickle.load(f)
    
    total_tasks = len(all_tasks)
    
    # 3. Write bash script
    with open(script_name, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Tile-Level Sensitivity Phase (Step 2)\n")
        f.write(f"# Targets: All 9 Tiles (9 Groups)\n\n")
        
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
    print(f"Total Tile Tasks: {total_tasks}")
    print(f"Tasks per Worker: {tasks_per_worker}")
    print(f"Total Worker Commands: {num_commands}")
    print(f"Results will be saved to: {out_dir}")

if __name__ == "__main__":
    main()
