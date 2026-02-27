## Usage Instructions

### 1. Data set creation
1. Run data/make_data_main.py to simulate data. 
2. Run models/train_models.py to train models.
3. Run data/shortlist_samples.py to shortlist samples for feature attribution.

---

### 2. SensX Analysis
1. **Global domain:** Run sensx_analysis/get_data_bounds.py.
2. Run sensx_analysis/generate_bash.py to create bash script for different hyperparameter configurations.
3. Run sensx_analysis/run_experiments.sh to run the SensX analysis. 

---

### 3. kernelSHAP analysis 
1. Run kSHAP_analysis/generate_bash_script_shap_analysis.py to create bash script for different hyperparameter configurations.
2. Run kSHAP_analysis/run_shap_experiments.sh to run the kernelSHAP analysis.

---

### 4. Integrated gradients analysis 
1. Run IG_analysis/generate_bash_script_IG_analysis.py to create bash script for different hyperparameter configurations.
2. Run IG_analysis/run_ig_experiments.sh to run the IG analysis.

