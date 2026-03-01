import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import glob

# --- Configuration ---
IMG_NAMES = ['000276', '000375']
MODEL_NAMES = ['Smiling', 'Eyeglasses']
N_W = 20
TOP_K_VALUES = [2500, 5000, 7500, 10000]
BORDER_WIDTH = 3
COLORMAP = 'jet'

SENSITIVITY_DIR = 'sensitivity'
DATA_DIR = 'data'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNEL_NAMES = ['R', 'G', 'B']

def load_and_average_sensitivity(img_name, model_name):
    """Load all batch files and average to get final sensitivity map."""
    batch_files = sorted(glob.glob(
        os.path.join(SENSITIVITY_DIR, f'sensx_{img_name}_{model_name}_nw{N_W}_batch*.npy')
    ))
    if len(batch_files) == 0:
        raise FileNotFoundError(f"No batch files found for {img_name}_{model_name}")
    print(f"  Found {len(batch_files)} batch files for {img_name} / {model_name}")
    
    arrays = [np.load(f) for f in batch_files]
    stacked = np.stack(arrays, axis=0)
    averaged = np.mean(stacked, axis=0)
    return averaged.squeeze()  # (3, 224, 224)

def make_mask(original_img_array, sensitivity, top_k):
    """Top-k features retain original values; rest set to 255 (white)."""
    C, H, W = sensitivity.shape
    sens_flat = sensitivity.reshape(-1)
    ranked_indices = np.argsort(-sens_flat)
    top_k_set = set(ranked_indices[:top_k])
    
    img_chw = original_img_array.transpose(2, 0, 1)
    mask = np.full_like(img_chw, 255, dtype=np.uint8)
    
    for idx in top_k_set:
        c = idx // (H * W)
        rem = idx % (H * W)
        h = rem // W
        w = rem % W
        mask[c, h, w] = img_chw[c, h, w]
    
    return mask.transpose(1, 2, 0)

def save_image(img_array, path):
    """Save a numpy array as an image with black border."""
    pil_img = Image.fromarray(img_array)
    pil_img = ImageOps.expand(pil_img, border=BORDER_WIDTH, fill='black')
    pil_img.save(path)
    print(f"  Saved {path}")

def load_original_image(img_name, size=(224, 224)):
    img_path = os.path.join(DATA_DIR, f'{img_name}.jpg')
    img = Image.open(img_path).convert('RGB').resize(size)
    return np.array(img)

def main():
    # --- Collect all sensitivities for global color scale ---
    all_sensitivities = {}
    for img_name in IMG_NAMES:
        for model_name in MODEL_NAMES:
            sensitivity = load_and_average_sensitivity(img_name, model_name)
            all_sensitivities[(img_name, model_name)] = sensitivity

    # Global color scale across all heatmaps
    global_vmin = min(s.min() for s in all_sensitivities.values())
    global_vmax = max(s.max() for s in all_sensitivities.values())

    # --- Save individual masks ---
    for img_name in IMG_NAMES:
        original_img = load_original_image(img_name)
        save_image(original_img, os.path.join(OUTPUT_DIR, f'{img_name}_original.png'))
        
        for model_name in MODEL_NAMES:
            sensitivity = all_sensitivities[(img_name, model_name)]
            for top_k in TOP_K_VALUES:
                mask_img = make_mask(original_img, sensitivity, top_k)
                fname = f'{img_name}_{model_name}_top{top_k}.png'
                save_image(mask_img, os.path.join(OUTPUT_DIR, fname))

    # --- Generate combined heatmap figure ---
    rows = []
    row_labels = []
    for img_name in IMG_NAMES:
        for model_name in MODEL_NAMES:
            rows.append((img_name, model_name))
            row_labels.append(f'{model_name}\n({img_name})')

    n_rows = len(rows)
    n_cols = 4  # Original + R + G + B

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for r, (img_name, model_name) in enumerate(rows):
        original_img = load_original_image(img_name)
        sensitivity = all_sensitivities[(img_name, model_name)]

        # Column 0: Original
        axes[r, 0].imshow(original_img)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=14, fontweight='bold')
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title('Original', fontsize=16, fontweight='bold')

        # Columns 1-3: R, G, B heatmaps
        for c, ch_name in enumerate(CHANNEL_NAMES):
            col = c + 1
            im = axes[r, col].imshow(sensitivity[c], cmap=COLORMAP, vmin=global_vmin, vmax=global_vmax)
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
            if r == 0:
                axes[r, col].set_title(f'{ch_name} Channel', fontsize=16, fontweight='bold')

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='SensX Attribution')

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    save_path = os.path.join(OUTPUT_DIR, 'vit_heatmaps_supplemental.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved supplemental heatmap figure to {save_path}")

if __name__ == '__main__':
    main()
