SensX: Model agnostic local feature attribution
=======
<img src="https://github.com/user-attachments/assets/2e57d376-089a-43a5-ba48-b7a48ccd9048" alt="SensX Sensitivity Map" width="400">

SensX identifies input features that a deep learning system is using to make its prediction.

SensX requires only the trained model. It does not need model internals, baseline references, and data the model was trained on.

It works for composite deep learing systems that depend on frozen heterogeneous components, API-only access, and high-dimensional inputs.



source ~/.bashrc

conda create -n sensx python=3.11 -y
conda install nomkl -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow requests datasets scikit-learn
pip install transformers[torch]
conda install matplotlib


For SHAP

conda create -n shap_env python=3.10 -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge shap matplotlib ipython

Vision Transformers:

1. Training

    1. Run download.py to download pretrained ViT model from HuggingFace.
    2. Download CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Make sure IMAGES_DIR is the location of the CelebA aligned images.
    3. Run process_celebA_data.py to generate the label file.
    3. Run finetune_vit.sh to train two binary classifiers, one to identify Smiling faces and other to identify Eyeglasses.



Run download.py





