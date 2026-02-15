## Usage Instructions

### 1. Training
1. **Dataset Acquisition:** Download single cell transcriptomics data of the Human Lung Cell Atlas (core) from https://datasets.cellxgene.cziscience.com/4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad (valid as of Feb 12, 2026). It has log-expression of ~27k genes for ~580k cells.
2. **Training:** Run `model/train.py` to train binary classifiers for different cell types.

---
**Shortlist cells for explanation:** Run shortlist_cells.py to get 1000 cells of each cell type that with 99% model prection of belong to that type.

### 2. SensX Analysis
1. **Global domain:** Run sensx_analysis/get_global_domain.py.
2.

---

### 3. Perturbation analysis to validate SensX

---

### 4. Plotting
*(Details to be added)*


