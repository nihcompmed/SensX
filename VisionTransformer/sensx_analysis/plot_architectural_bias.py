import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import glob
import os

# --- Configuration ---
IMG_NAMES = ['000276', '000375']
MODEL_NAMES = ['Smiling', 'Eyeglasses']
N_W = 20
PATCH_SIZE = 16
BORDER_WIDTH = 3
COLORMAP = 'jet'

SENSITIVITY_DIR = 'sensitivity'
DATA_DIR = 'data'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNEL_NAMES = ['R', 'G', 'B']
CHANNELS_FULL = ['Red', 'Green', 'Blue']
COLORS = ['tab:red', 'tab:green', 'tab:blue']

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

def median_normalize_patches(heatmap_3ch, patch_size=16):
    """
    Normalize each patch by its median, per channel.
    Returns (3, H, W) median-normalized map.
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
        
        # Reshape back to (H, W)
        norm_patches = norm_patches.reshape(num_patches_side, num_patches_side, patch_size, patch_size)
        norm_patches = norm_patches.transpose(0, 2, 1, 3).reshape(H, W)
        normalized[c] = norm_patches
    
    return normalized

def compute_patch_bias_profile(heatmap_3ch, patch_size=16):
    """
    Compute intra-patch bias profile for each channel.
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

def plot_patch_bias(unique_d, means, stds, save_path):
    """Plot a single patch bias profile and save."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for idx, (ch_name, color) in enumerate(zip(CHANNELS_FULL, COLORS)):
        ax.fill_between(unique_d, means[idx] - stds[idx], means[idx] + stds[idx],
                        color=color, alpha=0.15)
        ax.plot(unique_d, means[idx], '-o', color=color, markersize=4,
                linewidth=1.5, label=ch_name)
    
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Distance from Patch Center", fontsize=14)
    ax.set_ylabel("Patch-normalized SensX", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")

def main():
    # --- First pass: collect global ranges for consistent color scales ---
    all_sensitivities = {}
    for img_name in IMG_NAMES:
        for model_name in MODEL_NAMES:
            sensitivity = load_and_average_sensitivity(img_name, model_name)
            all_sensitivities[(img_name, model_name)] = sensitivity

    # Global color scale for raw heatmaps
    raw_vmin = min(s.min() for s in all_sensitivities.values())
    raw_vmax = max(s.max() for s in all_sensitivities.values())

    # Compute all normalized maps
    all_normalized = {}
    for key, sens in all_sensitivities.items():
        all_normalized[key] = median_normalize_patches(sens, PATCH_SIZE)

    ## Global color scale for normalized heatmaps
    #norm_vmin = min(n.min() for n in all_normalized.values())
    #norm_vmax = max(n.max() for n in all_normalized.values())

    norm_vmin = 0
    norm_vmax = 2.5

    # --- Second pass: save everything ---
    for img_name in IMG_NAMES:
        for model_name in MODEL_NAMES:
            print(f"Processing {img_name} / {model_name}...")
            sensitivity = all_sensitivities[(img_name, model_name)]
            normalized = all_normalized[(img_name, model_name)]

            # 1. Per-channel raw heatmaps (already saved by earlier script, but save again for completeness)
            for c, ch_name in enumerate(CHANNEL_NAMES):
                fname = f'{img_name}_{model_name}_heatmap_{ch_name}.png'
                save_heatmap(sensitivity[c], os.path.join(OUTPUT_DIR, fname), raw_vmin, raw_vmax)

            # 2. Per-channel median-normalized heatmaps
            for c, ch_name in enumerate(CHANNEL_NAMES):
                fname = f'{img_name}_{model_name}_normalized_{ch_name}.png'
                save_heatmap(normalized[c], os.path.join(OUTPUT_DIR, fname), norm_vmin, norm_vmax)

            # 3. Distance profile
            unique_d, means, stds = compute_patch_bias_profile(sensitivity, PATCH_SIZE)
            fname = f'patch_bias_{img_name}_{model_name}.png'
            plot_patch_bias(unique_d, means, stds, os.path.join(OUTPUT_DIR, fname))

if __name__ == '__main__':
    main()
