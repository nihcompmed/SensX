#!/bin/bash

(
CUDA_VISIBLE_DEVICES=0 python3 sensx_step3_sensitivity.py 24
CUDA_VISIBLE_DEVICES=0 python3 sensx_step3_sensitivity.py 18
CUDA_VISIBLE_DEVICES=0 python3 sensx_step3_sensitivity.py 12
CUDA_VISIBLE_DEVICES=0 python3 sensx_step3_sensitivity.py 6
CUDA_VISIBLE_DEVICES=0 python3 sensx_step3_sensitivity.py 0
) &

(
CUDA_VISIBLE_DEVICES=1 python3 sensx_step3_sensitivity.py 25
CUDA_VISIBLE_DEVICES=1 python3 sensx_step3_sensitivity.py 19
CUDA_VISIBLE_DEVICES=1 python3 sensx_step3_sensitivity.py 13
CUDA_VISIBLE_DEVICES=1 python3 sensx_step3_sensitivity.py 7
CUDA_VISIBLE_DEVICES=1 python3 sensx_step3_sensitivity.py 1
) &

(
CUDA_VISIBLE_DEVICES=2 python3 sensx_step3_sensitivity.py 26
CUDA_VISIBLE_DEVICES=2 python3 sensx_step3_sensitivity.py 20
CUDA_VISIBLE_DEVICES=2 python3 sensx_step3_sensitivity.py 14
CUDA_VISIBLE_DEVICES=2 python3 sensx_step3_sensitivity.py 8
CUDA_VISIBLE_DEVICES=2 python3 sensx_step3_sensitivity.py 2
) &

(
CUDA_VISIBLE_DEVICES=3 python3 sensx_step3_sensitivity.py 27
CUDA_VISIBLE_DEVICES=3 python3 sensx_step3_sensitivity.py 21
CUDA_VISIBLE_DEVICES=3 python3 sensx_step3_sensitivity.py 15
CUDA_VISIBLE_DEVICES=3 python3 sensx_step3_sensitivity.py 9
CUDA_VISIBLE_DEVICES=3 python3 sensx_step3_sensitivity.py 3
) &

(
CUDA_VISIBLE_DEVICES=4 python3 sensx_step3_sensitivity.py 28
CUDA_VISIBLE_DEVICES=4 python3 sensx_step3_sensitivity.py 22
CUDA_VISIBLE_DEVICES=4 python3 sensx_step3_sensitivity.py 16
CUDA_VISIBLE_DEVICES=4 python3 sensx_step3_sensitivity.py 10
CUDA_VISIBLE_DEVICES=4 python3 sensx_step3_sensitivity.py 4
) &

(
CUDA_VISIBLE_DEVICES=5 python3 sensx_step3_sensitivity.py 29
CUDA_VISIBLE_DEVICES=5 python3 sensx_step3_sensitivity.py 23
CUDA_VISIBLE_DEVICES=5 python3 sensx_step3_sensitivity.py 17
CUDA_VISIBLE_DEVICES=5 python3 sensx_step3_sensitivity.py 11
CUDA_VISIBLE_DEVICES=5 python3 sensx_step3_sensitivity.py 5
) &

wait
