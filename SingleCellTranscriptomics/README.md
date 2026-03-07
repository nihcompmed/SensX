## Usage Instructions

### 1. Training
1. **Dataset Acquisition:** Download single cell transcriptomics data of the Human Lung Cell Atlas (core) from https://datasets.cellxgene.cziscience.com/4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad (valid as of Feb 12, 2026). It has log-expression of ~27k genes for ~580k cells.
2. **Training:** Run `model/train.py` to train binary classifiers for different cell types.

---
**Shortlist cells for explanation:** Run shortlist_cells.py to get 1000 cells of each cell type that with 99% model prection of belong to that type.

### 2. SensX Analysis
1. **Global domain:** Run `sensx_analysis/get_global_domain.py.`
2. Run `sensx_analysis/sensx_step1_stability.py` for stability profile
3. Run `sensx_analysis/sensx_bash_v2.sh` for SensX values

---

### 3. Perturbation analysis to validate SensX
1. Run `sensx_analysis/perturbation_analysis.py`
2. Run `sensx_analysis/perturbation_analysis_delta1_sensx.sh`

---

### 4. Integrated Gradients variants analysis
1. Run `IG_analysis/analysis_bash.sh`

---

### 5. Perturbation analysis to validate IG
1. Run `IG_analysis/perturbation_analysis.py`
2. Run `IG_analysis/perturbation_analysis_delta1_sensx.sh`

---

### 6. DeepSHAP analysis
1. Run `deepSHAP_analysis/deep_shap.py`

---

### 7. Perturbation analysis to validate DeepSHAP
1. Run `deepSHAP_analysis/perturbation_analysis.py`
2. Run `deepSHAP_analysis/perturbation_analysis_delta1_sensx.sh`


