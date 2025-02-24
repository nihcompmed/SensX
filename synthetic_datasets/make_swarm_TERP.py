import pickle

n_runs = 100

all_n_tests = [10, 100, 1000]
all_n_samples = [100, 500, 1000, 2500]

dbfile = open('synthetic_data.p', 'rb')
data = pickle.load(dbfile)
dbfile.close()

swarm_ff = open('swarm_TERP.sh', 'w')

# TERP HAS TO run in sbatch and NOT swarm
swarm_ff.write('#!/bin/sh\n') 


for run_num in range(n_runs):

    for n_test in all_n_tests:

        for method_hyperpar in all_n_samples:

            for dataset in data:

                cmd = f'python3 terp_analysis.py {dataset} {n_test} {method_hyperpar} {run_num}\n'

                swarm_ff.write(cmd)


swarm_ff.close()



