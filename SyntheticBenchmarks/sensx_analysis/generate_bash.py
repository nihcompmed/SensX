import os

models = ['XOR', 'orange_skin', 'nonlinear_additive', 'switch']
n_ws = [500, 1000, 2500, 5000, 7500, 10000]
n_runs = 100

output_file = 'run_experiments.sh'

with open(output_file, 'w') as f:
    #f.write('#!/bin/bash\n\n')

    for run_num in range(1, n_runs + 1):
        for model in models:
            for n_w in n_ws:
                f.write(f'python3 sensx_worker.py {model} {n_w} {run_num}\n')

print(f"Generated {output_file} with {len(models) * len(n_ws) * n_runs} commands.")
