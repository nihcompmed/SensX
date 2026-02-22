#!/bin/bash

for n_steps in 100 200 500; do
    echo "========== n_steps = $n_steps =========="

    # Zero + mean baselines
    python3 ig_vit.py deterministic $n_steps

    # Expected gradients (100 random baselines averaged)
    python3 ig_vit.py expected_gradients $n_steps
done

echo "All done."

