
1. Training
    a. Run download.py to download pretrained ViT model from HuggingFace.
    b. Download CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Make sure IMAGES_DIR is the location of the CelebA aligned images.
    c. Run process\_celebA\_data.py to generate the label file.
    d. Run finetune\_vit.sh to train two binary classifiers---one will predict Smiling faces and the other will predict Eyeglasses.

2. SensX Analysis 
    a. (Optional) Edit SensX hyperparameters in sensx\_step1\_stability.py.
    b. Run sensx\_step1\_stability.py it to get the stability profile.
    c. (optional) Edit SensX hyperparameters in sensx\_step3\_sensitivity.py.
    d. Run sensx\_bash\_v2.sh to get the SensX values of batches.

3. Plotting







