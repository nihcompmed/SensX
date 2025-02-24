1. Get global domain: Run `get_hypercube.py`. Input file1 from finetuning. Output `f'category{do_cat}_timestamp{ts}_hypercube.npz'`.
2. Compute $\delta_star$ for SensX: Run `sensex_analysis_delta_star.py`.
2. Compute SensX values: Run `make_swarm_sensex.py` and run the resulting bash script file.
