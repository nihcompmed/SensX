"""
Changes from Previous Version:
- Initial version of the bash generator for the Step 3 Perturbation Test.
- Sets tasks_per_worker to 10 to prevent long-running jobs from timing out on the scheduler.
"""

import pickle
import os
import numpy as np

def main():
    # --- CONFIGURATION ---
    out_dir = 'perturbation_test_results'
    
    # Each task involves 2,000 forward passes (1000 center, 1000 neighbors).
    # We keep tasks per worker low to manage wall-time and VRAM usage.
    tasks_per_worker = 100 
    
    script_name = 'run_step3_perturbation_test.sh'
    
    TASK_FILE = 'tile_sensitivity_tasks.p'
    PYTHON_EXEC = 'python3'
    WORKER_SCRIPT = 'step3_perturbation_test_worker.py'

    if not os.path.exists(out_dir):
        print(f"Creating perturbation results directory: {out_dir}")
        os.makedirs(out_dir)

    if not os.path.exists(TASK_FILE):
        print(f"Error: {TASK_FILE} not found. Ensure aggregate_tile_stability.py has been run.")
        return

    with open(TASK_FILE, 'rb') as f:
        all_tasks = pickle.load(f)
    
    total_tasks = len(all_tasks)
    
    with open(script_name, 'w') as f:
        #f.write("#!/bin/bash\n\n")
        #f.write(f"# Step 3: Empirical Perturbation Test (Ablation Study)\n")
        #f.write(f"# Distribution: 1000 Center vs 1000 Neighbor samples per task\n\n")
        
        for start in range(0, total_tasks, tasks_per_worker):
            end = min(start + tasks_per_worker, total_tasks)
            
            # Construct command
            cmd = (
                f"{PYTHON_EXEC} {WORKER_SCRIPT} "
                f"--task_file {TASK_FILE} "
                f"--task_start {start} "
                f"--task_end {end} "
                f"--out_dir {out_dir}"
            )
            f.write(cmd + "\n")

    os.chmod(script_name, 0o755)
    num_commands = int(np.ceil(total_tasks / tasks_per_worker))
    
    print(f"Successfully generated {script_name}")
    print(f"Total Perturbation Tasks: {total_tasks}")
    print(f"Tasks per Worker: {tasks_per_worker}")
    print(f"Total Worker Commands (Job Array Size): {num_commands}")
    print("-" * 30)

if __name__ == "__main__":
    main()
