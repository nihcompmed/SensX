import random

# Configuration
datasets = ['XOR', 'orange_skin', 'nonlinear_additive', 'switch']
baseline_types = ['zero', 'mean', 'random']
n_steps_list = [50, 100, 200, 500]
n_runs_random = 100  # Only for random baseline (sampling-based)
script_name = "ig_worker.py"
output_bash_file = "run_ig_experiments.sh"

def generate_script():
    commands = []

    for dataset in datasets:
        for n_steps in n_steps_list:
            for baseline_type in baseline_types:
                if baseline_type in ('zero', 'mean'):
                    # Deterministic — single run, no run_number needed
                    cmd = f"python3 {script_name} {dataset} {n_steps} {baseline_type}"
                    commands.append(f"{cmd}\n")
                elif baseline_type == 'random':
                    # Sampling-based — repeat like SHAP
                    for run_id in range(1, n_runs_random + 1):
                        cmd = f"python3 {script_name} {dataset} {n_steps} {baseline_type} {run_id}"
                        commands.append(f"{cmd}\n")

    # Shuffle to distribute workload
    random.shuffle(commands)

    with open(output_bash_file, 'w') as f:
        for cmd_block in commands:
            f.write(cmd_block)

    n_deterministic = len(datasets) * len(n_steps_list) * 2
    n_random = len(datasets) * len(n_steps_list) * n_runs_random
    print(f"Generated {output_bash_file} with {len(commands)} commands.")
    print(f"  Deterministic (zero + mean): {n_deterministic}")
    print(f"  Sampling-based (random): {n_random}")
    print(f"  n_steps values: {n_steps_list}")
    print(f"Run it with: bash {output_bash_file}")

if __name__ == "__main__":
    generate_script()
