# SensX
SensX PyTorch

source ~/.bashrc

conda create -n sensx python=3.11 -y
conda install nomkl -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow requests datasets scikit-learn
pip install transformers[torch]
conda install matplotlib

Vision Transformers:

1. Training

    1. Run download.py to download pretrained ViT model from HuggingFace.
    2. Download CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Make sure IMAGES_DIR is the location of the CelebA aligned images.
    3. Run process_celebA_data.py to generate the label file.
    3. Run finetune_vit.sh to train two binary classifiers, one to identify Smiling faces and other to identify Eyeglasses.



Run download.py





