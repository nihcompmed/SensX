## Usage Instructions

### 1. Training
1. **Download Model:** Run `model/download.py` to download the pretrained ViT model from HuggingFace.
2. **Dataset Acquisition:** Download the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 
   > **Note:** Ensure `IMAGES_DIR` is set to the location of the CelebA aligned images.
3. **Preprocessing:** Run `model/process_celebA_data.py` to generate the label file.
4. **Fine-tuning:** Run `model/finetune_vit.sh` to train two binary classifiers, one to predict smiling faces and the other to predict eyeglasses.

---

### 2. SensX Analysis
1. **Stability Profile:**
   * *(Optional)* Edit hyperparameters in `sensx_analysis/sensx_step1_stability.py`.
   * Run `sensx_analysis/sensx_step1_stability.py` to generate the profile.
2. **Sensitivity Analysis:**
   * *(Optional)* Edit hyperparameters in `sensx_analysis/sensx_step3_sensitivity.py`.
   * Run `sensx_bash_v2.sh` to compute the SensX values of batches.

---

### 3. Perturbation analysis to validate SensX
1. Run `sensx_analysis/perturbation_analysis.py`.

---

### 4. Adebayo check to validate SensX
1. Run `sensx_analysis/adebayo_analysis/create_adebayo_models.py`

---

### 5. Plotting
*(Details to be added)*


