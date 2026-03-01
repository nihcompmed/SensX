## Usage Instructions

### 1. Training
1. **Download Model:** `model/download.py` to download the pretrained ViT model from HuggingFace.
2. **Dataset Acquisition:** Download the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 
   > **Note:** Ensure `IMAGES_DIR` is set to the location of the CelebA aligned images.
3. **Preprocessing:** `model/process_celebA_data.py` to generate the label file.
4. **Fine-tuning:** `model/finetune_vit.sh` to train two binary classifiers, one to predict smiling faces and the other to predict eyeglasses.

---

### 2. SensX Analysis
1. **Stability Profile:**
   * *(Optional)* Edit hyperparameters in `sensx_analysis/sensx_step1_stability.py`.
   * `sensx_analysis/sensx_step1_stability.py` to generate the profile.
2. **Sensitivity Analysis:**
   * *(Optional)* Edit hyperparameters in `sensx_analysis/sensx_step3_sensitivity.py`.
   * `sensx_analysis/sensx_bash_v2.sh` to compute the SensX values of batches.
3. **Plotting:**
   * `sensx_analysis/plot_masks_fig1.py` for masks of top SensX ranks and heatmaps.
   * `sensx_analysis/plot_stability_profile.py` for stability profiles.
   * `sensx_analysis/plot_architectural_bias.py` for patch-normalized maps and distance from center bias.
4. **Perturbation analysis for validation:**
   * `sensx_analysis/perturbation_analysis.py`

---

### 4. Adebayo check to validate SensX
1. **Create cascading randomized models:**
    * `sensx_analysis/adebayo_analysis/create_adebayo_models.py`
2. **SensX analysis:**
    * `sensx_analysis/adebayo_analysis/sensx_step1_stability.py`
    * `sensx_analysis/adebayo_analysis/sensx_step3_sensitivity.py`
    * `sensx_analysis/adebayo_analysis/sensx_bash_v2.sh`
3. **Plotting:** `sensx_analysis/plot_adebayo_check.py`

---

### 5. IG analysis

1. `IG_analysis/run_ig_vit.sh`
2. `IG_analysis/plot_ig_vit.py` to plot masks of top IG ranks and heatmaps.
2. `IG_analysis/architectural_bias.py` for patch-normalized maps and distance from center bias.
 



