
datasets = ['Smiling', 'Eyeglasses']

swarm_ff = open('swarm_sensex.sh', 'w')

for batch in range(50):

    for dataset in datasets:

        cmd = f'python3 sensex_analysis_sensi.py {dataset} 10 {batch}\n'

        swarm_ff.write(cmd)

swarm_ff.close()


