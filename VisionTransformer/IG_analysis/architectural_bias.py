"""
Intra-patch analysis for IG attributions on ViT.
Uses identical methodology as plot_architectural_bias.py (SensX).

Usage:
    python3 intra_patch_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
IMAGES = ['000276', '000375']
MODELS = ['Smiling', 'Eyeglasses']
BASELINES = ['zero', 'mean', 'expected_gradients']
BASELINE_LABELS = {'zero': 'IG (zero)', 'mean': 'IG (mean)', 'expected_gradients': 'IG (expected gradients)'}
RESULTS_DIR = 'ig_results_ns500'
OUTPUT_DIR = 'ig_figures'
PATCH_SIZE = 16

from PIL import Image, ImageOps

COLORMAP = 'jet'
BORDER_WIDTH = 3


def median_normalize_patches(heatmap_3ch, patch_size=PATCH_SIZE):
    """
    Normalize each patch by its median, per channel.
    Identical to SensX plot_architectural_bias.py.
    """
    C, H, W = heatmap_3ch.shape
    num_patches_side = H // patch_size
    normalized = np.zeros_like(heatmap_3ch)

    for c in range(C):
        ch_data = heatmap_3ch[c]
        patches = ch_data.reshape(num_patches_side, patch_size, num_patches_side, patch_size)
        patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

        patch_medians = np.median(patches, axis=(1, 2), keepdims=True)
        norm_patches = np.divide(patches, patch_medians,
                                 out=np.zeros_like(patches),
                                 where=patch_medians != 0)

        norm_patches = norm_patches.reshape(num_patches_side, num_patches_side, patch_size, patch_size)
        norm_patches = norm_patches.transpose(0, 2, 1, 3).reshape(H, W)
        normalized[c] = norm_patches

    return normalized


def save_heatmap(data_2d, path, vmin, vmax, cmap=COLORMAP):
    """Save a 2D array as a colormapped heatmap with black border."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(data_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    pil_img = Image.open(path)
    pil_img = ImageOps.expand(pil_img, border=BORDER_WIDTH, fill='black')
    pil_img.save(path)
    print(f"  Saved {path}")


CHANNEL_NAMES = ['R', 'G', 'B']
CHANNELS_FULL = ['Red', 'Green', 'Blue']
COLORS = ['tab:red', 'tab:green', 'tab:blue']


def compute_patch_bias_profile(heatmap_3ch, patch_size=PATCH_SIZE):
    """
    Compute intra-patch bias profile for each channel.
    Identical to SensX plot_architectural_bias.py.
    """
    num_patches_side = heatmap_3ch.shape[1] // patch_size

    y, x = np.indices((patch_size, patch_size))
    center = (patch_size - 1) / 2.0
    dist_map = np.maximum(np.abs(x - center), np.abs(y - center))
    unique_d = np.unique(dist_map)

    all_means = []
    all_stds = []

    for idx in range(3):
        ch_data = heatmap_3ch[idx]
        patches = ch_data.reshape(num_patches_side, patch_size, num_patches_side, patch_size)
        patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

        patch_medians = np.median(patches, axis=(1, 2), keepdims=True)
        norm_patches = np.divide(patches, patch_medians,
                                 out=np.zeros_like(patches),
                                 where=patch_medians != 0)

        ch_means = []
        ch_stds = []

        for d in unique_d:
            mask = (dist_map == d)
            shell_values = norm_patches[:, mask]
            patch_shell_means = shell_values.mean(axis=1)

            valid_means = patch_shell_means[np.isfinite(patch_shell_means) & (patch_shell_means > 0)]

            if len(valid_means) > 0:
                ch_means.append(np.mean(valid_means))
                ch_stds.append(np.std(valid_means))
            else:
                ch_means.append(np.nan)
                ch_stds.append(np.nan)

        all_means.append(np.array(ch_means))
        all_stds.append(np.array(ch_stds))

    return unique_d, np.array(all_means), np.array(all_stds)


def plot_patch_bias(unique_d, means, stds, save_path, ylabel="Patch-normalized |IG|\n(Multiple of Patch Median)"):
    """
    Plot a single patch bias profile and save.
    Identical layout to SensX plot_architectural_bias.py.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, (ch_name, color) in enumerate(zip(CHANNELS_FULL, COLORS)):
        ax.fill_between(unique_d, means[idx] - stds[idx], means[idx] + stds[idx],
                        color=color, alpha=0.15)
        ax.plot(unique_d, means[idx], '-o', color=color, markersize=4,
                linewidth=1.5, label=ch_name)

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Distance from Patch Center (Chebyshev)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- First pass: collect all abs attribution maps and compute global ranges ---
    all_abs_attrs = {}
    all_normalized = {}

    for img_name in IMAGES:
        for model_name in MODELS:
            for bl in BASELINES:
                path = os.path.join(RESULTS_DIR, f'ig_{img_name}_{model_name}_{bl}.npy')
                if not os.path.exists(path):
                    print(f"  WARNING: missing {path}")
                    continue

                abs_attr = np.abs(np.load(path))  # (3, 224, 224)
                key = (img_name, model_name, bl)
                all_abs_attrs[key] = abs_attr
                all_normalized[key] = median_normalize_patches(abs_attr)

    if not all_abs_attrs:
        print("No data found!")
        return

    # Global color scales
    raw_vmin = min(s.min() for s in all_abs_attrs.values())
    raw_vmax = max(s.max() for s in all_abs_attrs.values())
    norm_vmin = 0
    norm_vmax = 2.5

    # --- Second pass: save everything ---
    for img_name in IMAGES:
        for model_name in MODELS:
            for bl in BASELINES:
                key = (img_name, model_name, bl)
                if key not in all_abs_attrs:
                    continue

                print(f"\n{img_name} x {model_name} x {bl}:")
                abs_attr = all_abs_attrs[key]
                normalized = all_normalized[key]

                # 1. Per-channel raw heatmaps
                for c, ch_name in enumerate(CHANNEL_NAMES):
                    fname = f'{img_name}_{model_name}_{bl}_heatmap_{ch_name}.png'
                    save_heatmap(abs_attr[c], os.path.join(OUTPUT_DIR, fname), raw_vmin, raw_vmax)

                # 2. Per-channel median-normalized heatmaps
                for c, ch_name in enumerate(CHANNEL_NAMES):
                    fname = f'{img_name}_{model_name}_{bl}_normalized_{ch_name}.png'
                    save_heatmap(normalized[c], os.path.join(OUTPUT_DIR, fname), norm_vmin, norm_vmax)

                # 3. Distance profile
                unique_d, means, stds = compute_patch_bias_profile(abs_attr)
                save_path = os.path.join(OUTPUT_DIR, f'patch_bias_{img_name}_{model_name}_{bl}.png')
                plot_patch_bias(unique_d, means, stds, save_path)


if __name__ == "__main__":
    main()
