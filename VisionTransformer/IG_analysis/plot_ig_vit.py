"""
Plot IG top-k feature masks for ViT analysis.
Formatting matched to SensX mask figures (heavy black borders, large labels).

Produces one figure per (image, model) combination with:
  - Rows: IG baselines (Zero, Mean, Expected Gradients)
  - Columns: Top-2500, Top-7500
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import os

# --- Configuration ---
IMAGES = ['000276', '000375']
MODELS = ['Smiling', 'Eyeglasses']
BASELINES = ['zero', 'mean', 'expected_gradients']
BASELINE_LABELS = {
    'zero': 'IG (zero)',
    'mean': 'IG (mean)',
    'expected_gradients': 'Expected\nGradients',
}
RESULTS_DIR = 'ig_results_ns500'
IMAGE_DIR = 'data'
OUTPUT_DIR = 'ig_figures'
TOP_K_VALUES = [2500, 7500]

# --- Font sizes matched to SensX figure ---
FS_TITLE = 24
FS_COL_HEADER = 40
FS_ROW_LABEL = 36
BORDER_WIDTH = 3.0  # points, for spines


def load_original_image(img_name):
    """Load and resize original image to 224x224."""
    img_path = os.path.join(IMAGE_DIR, f'{img_name}.jpg')
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    return np.array(img)  # (224, 224, 3), uint8


def create_mask(orig_img, abs_attrs_flat, top_k):
    """Create a white image with top-k features filled from original."""
    masked = np.full_like(orig_img, 255)  # (224, 224, 3)
    top_k_indices = np.argsort(abs_attrs_flat)[-top_k:]

    for idx in top_k_indices:
        c = idx // (224 * 224)
        remainder = idx % (224 * 224)
        r = remainder // 224
        col_px = remainder % 224
        masked[r, col_px, c] = orig_img[r, col_px, c]

    return masked


def plot_masks(img_name, model_name):
    """
    Plot top-k feature masks: rows = IG baselines, columns = top-k values.
    Formatting matched to SensX mask figure style.
    """
    orig_img = load_original_image(img_name)

    n_rows = len(BASELINES)
    n_cols = len(TOP_K_VALUES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5.5 * n_rows),
        gridspec_kw={'wspace': 0.08, 'hspace': 0.12},
    )

    # Ensure 2D array of axes
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, bl in enumerate(BASELINES):
        path = os.path.join(RESULTS_DIR, f'ig_{img_name}_{model_name}_{bl}.npy')
        if not os.path.exists(path):
            for col in range(n_cols):
                axes[row, col].axis('off')
            continue

        attrs = np.load(path)  # (3, 224, 224)
        abs_attrs = np.abs(attrs).flatten()  # 150528

        for col, top_k in enumerate(TOP_K_VALUES):
            ax = axes[row, col]

            masked = create_mask(orig_img, abs_attrs, top_k)
            ax.imshow(masked)
            ax.set_xticks([])
            ax.set_yticks([])

            # Heavy black border on all spines
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(BORDER_WIDTH)
                spine.set_edgecolor('black')

            # Column headers (top row only)
            if row == 0:
                ax.set_title(
                    f'Top {top_k:,}',
                    fontsize=FS_COL_HEADER,
                    pad=12,
                )

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel(
                    BASELINE_LABELS[bl],
                    fontsize=FS_ROW_LABEL,
                    labelpad=10,
                )

    ## Figure title
    #fig.suptitle(
    #    f'{img_name} — {model_name}',
    #    fontsize=FS_TITLE,
    #    fontweight='bold',
    #    y=1.02,
    #)

    save_path = os.path.join(OUTPUT_DIR, f'mask_{img_name}_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def plot_heatmaps(img_name, model_name):
    """
    Plot |IG| heatmaps: rows = baselines, columns = R/G/B channels.
    Formatting matched to SensX style.
    """
    CHANNEL_NAMES = ['Red', 'Green', 'Blue']

    n_rows = len(BASELINES)
    n_cols = 3

    # Collect data and find global vmax
    all_abs = {}
    for bl in BASELINES:
        path = os.path.join(RESULTS_DIR, f'ig_{img_name}_{model_name}_{bl}.npy')
        if os.path.exists(path):
            all_abs[bl] = np.abs(np.load(path))
    if not all_abs:
        print(f"  No data found for {img_name}_{model_name}")
        return

    vmax = max(a.max() for a in all_abs.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5.5 * n_rows),
        gridspec_kw={'wspace': 0.08, 'hspace': 0.12},
    )

    for row, bl in enumerate(BASELINES):
        if bl not in all_abs:
            for col in range(n_cols):
                axes[row, col].axis('off')
            continue

        attrs = all_abs[bl]  # (3, 224, 224)

        for col in range(n_cols):
            ax = axes[row, col]
            im = ax.imshow(attrs[col], cmap='jet', vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(BORDER_WIDTH)
                spine.set_edgecolor('black')

            if row == 0:
                ax.set_title(
                    CHANNEL_NAMES[col],
                    fontsize=FS_COL_HEADER,
                    fontweight='bold',
                    pad=12,
                )
            if col == 0:
                ax.set_ylabel(
                    BASELINE_LABELS[bl],
                    fontsize=FS_ROW_LABEL,
                    fontweight='bold',
                    labelpad=10,
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('|IG attribution|', fontsize=FS_ROW_LABEL)
    cbar.ax.tick_params(labelsize=14)

    fig.suptitle(
        f'{img_name} — {model_name}',
        fontsize=FS_TITLE,
        fontweight='bold',
        y=1.02,
    )

    save_path = os.path.join(OUTPUT_DIR, f'heatmap_{img_name}_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_name in IMAGES:
        for model_name in MODELS:
            print(f"\n{img_name} x {model_name}:")
            plot_masks(img_name, model_name)
            plot_heatmaps(img_name, model_name)


if __name__ == "__main__":
    main()
