import numpy as np
import glob
import os
import re
from collections import defaultdict

# --- Configuration ---
SENSITIVITY_DIR = 'sensitivity'
OUTPUT_DIR = 'sensitivity_aggregated'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Discover and group files by cell type ---
all_files = glob.glob(os.path.join(SENSITIVITY_DIR, 'sensx_*_nw*.npy'))

if not all_files:
    raise FileNotFoundError(f"No sensitivity files found in {SENSITIVITY_DIR}/")

pattern = re.compile(r'^sensx_(.+)_nw(\d+)_batch(\d+)\.npy$')

files_by_ctype = defaultdict(list)

for fpath in all_files:
    fname = os.path.basename(fpath)
    m = pattern.match(fname)
    if m:
        ctype = m.group(1)
        batch = int(m.group(3))
        files_by_ctype[ctype].append((batch, fpath))
    else:
        print(f"WARNING: Could not parse filename: {fname}")

print(f"Found {len(files_by_ctype)} cell types.\n")

# --- 2. Average over batches per cell type, save per-cell values ---
ctype_names = sorted(files_by_ctype.keys())

for ctype in ctype_names:
    batch_files = sorted(files_by_ctype[ctype], key=lambda x: x[0])
    n_batches = len(batch_files)

    first = np.load(batch_files[0][1])
    print(f"{ctype}: {n_batches} batches, shape = {first.shape}")

    all_batches = np.zeros((n_batches, *first.shape), dtype=np.float64)
    all_batches[0] = first

    for i, (batch_id, fpath) in enumerate(batch_files[1:], start=1):
        all_batches[i] = np.load(fpath)

    # Average over batches (axis=0), keep per-cell values
    percell = np.mean(all_batches, axis=0)  # (N_cells, 1, N_genes)
    percell = percell.squeeze()  # (N_cells, N_genes)

    out_path = os.path.join(OUTPUT_DIR, f'sensx_percell_{ctype}.npy')
    np.save(out_path, percell.astype(np.float32))
    print(f"  Saved {percell.shape} to {out_path}")

# Save cell type order
with open(os.path.join(OUTPUT_DIR, 'cell_type_order.txt'), 'w') as f:
    for i, ct in enumerate(ctype_names):
        f.write(f"{i}\t{ct}\n")

print(f"\nDone. Saved per-cell SensX values for {len(ctype_names)} cell types.")
