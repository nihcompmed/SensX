import random

# Configuration
datasets = ['XOR', 'orange_skin', 'nonlinear_additive', 'switch']
nsamples_list = [500, 1000, 2500, 5000, 7500, 10000]
n_runs = 100
script_name = "shap_worker.py"
output_bash_file = "run_shap_experiments.sh"

def generate_script():
    commands = []

    # 1. Generate all command combinations
    for dataset in datasets:
        for nsamples in nsamples_list:
            for run_id in range(1, n_runs + 1):
                # Construct command: python3 shap_worker.py <dataset> <nsamples> <run_number>
                cmd = f"python3 {script_name} {dataset} {nsamples} {run_id}"
                
                full_block = f"{cmd}\n"
                commands.append(full_block)

    # 2. Shuffle the commands to distribute workload
    random.shuffle(commands)

    # 3. Write to file
    with open(output_bash_file, 'w') as f:
        for cmd_block in commands:
            f.write(cmd_block)
                    
    print(f"Generated {output_bash_file} with {len(commands)} randomized commands.")
    print(f"Run it with: bash {output_bash_file}")

if __name__ == "__main__":
    generate_script()
