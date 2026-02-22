CUDA_VISIBLE_DEVICES=0 python3 sensx_step3_sensitivity.py 0 & 
CUDA_VISIBLE_DEVICES=1 python3 sensx_step3_sensitivity.py 1 &
CUDA_VISIBLE_DEVICES=2 python3 sensx_step3_sensitivity.py 2 &
CUDA_VISIBLE_DEVICES=3 python3 sensx_step3_sensitivity.py 3 &
CUDA_VISIBLE_DEVICES=4 python3 sensx_step3_sensitivity.py 4 &
CUDA_VISIBLE_DEVICES=5 python3 sensx_step3_sensitivity.py 5 &
wait
