import torch
import numpy as np
import os
import sys
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from captum.attr import IntegratedGradients

# --- Command Line Arguments ---
# Usage:
#   python3 ig_vit.py deterministic <n_steps>
#   python3 ig_vit.py expected_gradients <n_steps>
baseline_mode = sys.argv[1] if len(sys.argv) > 1 else 'deterministic'
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

# --- Configuration ---
MODELS = {
    'Smiling': '../model/vit-Smiling-model-final/',
    'Eyeglasses': '../model/vit-Eyeglasses-model-final/',
}
IMAGES = ['000276', '000375']
IMAGE_DIR = '../model/data'
OUTPUT_DIR = f'ig_results_ns{N_STEPS}'
MEAN_IMAGE_PATH = 'celeba_train_mean.npy'
CELEBA_PICKLE = '../model/CelebA_img_labels.p'
CELEBA_IMAGES_DIR = '../model/CelebA/img_align_celeba'
N_RANDOM_RUNS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_expected_gradients():
    """Compute Expected Gradients in one pass: average IG over N_RANDOM_RUNS random baselines."""
    import pickle as pkl
    from datasets import Dataset

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Computing Expected Gradients (n_steps={N_STEPS}, {N_RANDOM_RUNS} baselines)...")

    # Load training image paths once
    with open(CELEBA_PICKLE, 'rb') as f:
        label_dict = pkl.load(f)
    all_paths = []
    for fn in label_dict.keys():
        fp = os.path.join(CELEBA_IMAGES_DIR, fn)
        if os.path.exists(fp):
            all_paths.append(fp)

    ds = Dataset.from_dict({"image": all_paths, "idx": list(range(len(all_paths)))})
    split = ds.train_test_split(test_size=0.15, seed=42)
    train_paths = [all_paths[i] for i in split["train"]["idx"]]
    print(f"Training images: {len(train_paths)}")

    if len(train_paths) == 0:
        print("Error: No training images found. Check CELEBA_IMAGES_DIR.")
        return

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*10} {model_name} model {'='*10}")

        model = ViTForImageClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(DEVICE)
        model.eval()

        processor = ViTImageProcessor.from_pretrained(model_path)

        def forward_fn(pixel_values):
            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[:, 1]

        ig = IntegratedGradients(forward_fn)

        for img_name in IMAGES:
            save_path = os.path.join(OUTPUT_DIR, f'ig_{img_name}_{model_name}_expected_gradients.npy')
            if os.path.exists(save_path):
                print(f"  Skipping {img_name}: {save_path} exists.")
                continue

            img_path = os.path.join(IMAGE_DIR, f'{img_name}.jpg')
            raw_image = Image.open(img_path).convert("RGB")
            inputs = processor(images=raw_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(DEVICE)
            pixel_values.requires_grad_(True)

            print(f"\n  {img_name}: averaging over {N_RANDOM_RUNS} random baselines...")
            running_sum = None

            for r in range(1, N_RANDOM_RUNS + 1):
                rng = np.random.default_rng(seed=r)
                rand_idx = rng.integers(0, len(train_paths))
                rand_img = Image.open(train_paths[rand_idx]).convert("RGB")
                rand_inputs = processor(images=rand_img, return_tensors="pt")
                rand_baseline = rand_inputs['pixel_values'].to(DEVICE)

                attrs = ig.attribute(
                    pixel_values,
                    baselines=rand_baseline,
                    n_steps=N_STEPS,
                    method='gausslegendre',
                )

                attrs_np = attrs.detach().cpu().numpy().squeeze(0)

                if running_sum is None:
                    running_sum = attrs_np.astype(np.float64)
                else:
                    running_sum += attrs_np.astype(np.float64)

                if r % 10 == 0:
                    print(f"    {r}/{N_RANDOM_RUNS} done")

            avg_map = (running_sum / N_RANDOM_RUNS).astype(np.float32)
            np.save(save_path, avg_map)
            print(f"  Saved to {save_path}")

        del model, ig
        torch.cuda.empty_cache()

    print("\nExpected Gradients complete.")


def main():
    if baseline_mode == 'expected_gradients':
        compute_expected_gradients()
        return

    if baseline_mode != 'deterministic':
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}. Use 'deterministic' or 'expected_gradients'.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*10} Loading {model_name} model {'='*10}")

        model = ViTForImageClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(DEVICE)
        model.eval()

        processor = ViTImageProcessor.from_pretrained(model_path)

        def forward_fn(pixel_values):
            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[:, 1]

        ig = IntegratedGradients(forward_fn)

        # Zero and mean baselines
        imagenet_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1).to(DEVICE)
        imagenet_std = torch.tensor(processor.image_std).view(1, 3, 1, 1).to(DEVICE)
        zero_baseline = (-imagenet_mean / imagenet_std).expand(1, 3, 224, 224).clone()

        mean_image = np.load(MEAN_IMAGE_PATH)
        mean_baseline = torch.from_numpy(mean_image).unsqueeze(0).to(DEVICE)

        baselines = [('zero', zero_baseline), ('mean', mean_baseline)]

        # --- Loop over images ---
        for img_name in IMAGES:
            img_path = os.path.join(IMAGE_DIR, f'{img_name}.jpg')
            print(f"\n  Processing {img_name} with {model_name} model...")

            raw_image = Image.open(img_path).convert("RGB")
            inputs = processor(images=raw_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(DEVICE)
            pixel_values.requires_grad_(True)

            with torch.no_grad():
                pred = forward_fn(pixel_values)
                print(f"  Prediction P({model_name}): {pred.item():.4f}")

            for baseline_name, baseline in baselines:
                # Skip if already computed
                save_path = os.path.join(
                    OUTPUT_DIR,
                    f'ig_{img_name}_{model_name}_{baseline_name}.npy'
                )
                if os.path.exists(save_path):
                    print(f"    Skipping {baseline_name}: {save_path} exists.")
                    continue

                print(f"    Computing IG with {baseline_name} baseline, n_steps={N_STEPS}...")

                attrs, delta = ig.attribute(
                    pixel_values,
                    baselines=baseline,
                    n_steps=N_STEPS,
                    method='gausslegendre',
                    return_convergence_delta=True,
                )

                attrs_np = attrs.detach().cpu().numpy().squeeze(0)  # (3, 224, 224)
                delta_val = delta.detach().cpu().item()

                print(f"    Convergence delta: {delta_val:.6f}")
                print(f"    Attribution shape: {attrs_np.shape}")
                print(f"    Attribution range: [{attrs_np.min():.6f}, {attrs_np.max():.6f}]")

                np.save(save_path, attrs_np)
                print(f"    Saved to {save_path}")

            # Save preprocessed input once
            input_save = os.path.join(OUTPUT_DIR, f'input_{img_name}.npy')
            if not os.path.exists(input_save):
                np.save(input_save, pixel_values.detach().cpu().numpy().squeeze(0))

        del model, ig
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()

