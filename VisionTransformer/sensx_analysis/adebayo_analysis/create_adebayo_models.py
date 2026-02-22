"""
Adebayo Cascading Randomization for ViT-base
=============================================
Creates progressively randomized versions of a fine-tuned ViT model
following the cascading randomization protocol of Adebayo et al. (2018).

Weights are reinitialized using each layer's default reset_parameters()
method (matching the original paper's approach). The classification head
is left intact at all levels — randomizing it is trivially uninformative.

Usage:
    python create_adebayo_models.py --model_path /path/to/finetuned_model \
                                     --output_dir /path/to/output

Each cascade level saves a full model loadable via:
    ViTForImageClassification.from_pretrained(level_path)
"""

import argparse
import os
import copy
import torch
from transformers import ViTForImageClassification, ViTImageProcessor


# ── Cascade levels ──────────────────────────────────────────────────────
# Each entry: (level_name, description, encoder_blocks_to_randomize, randomize_embeddings)
CASCADE_LEVELS = [
    ("level_0_original",    "Original model (no randomization)",             [],               False),
    ("level_1_block11",     "Top encoder block randomized (block 11)",       [11],             False),
    ("level_2_blocks8to11", "Top third randomized (blocks 8-11)",            [8, 9, 10, 11],   False),
    ("level_3_blocks6to11", "Top half randomized (blocks 6-11)",             [6, 7, 8, 9, 10, 11], False),
    ("level_4_blocks0to11", "All encoder blocks randomized (blocks 0-11)",   list(range(12)),   False),
    ("level_5_all",         "Full randomization (encoder + embeddings)",      list(range(12)),   True),
]


def reset_module_parameters(module):
    """
    Recursively reset parameters of a module using its default initialization.
    Falls back to xavier_uniform for weights and zeros for biases if
    reset_parameters() is not available.
    """
    # If the module itself has reset_parameters, use it
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
        return

    # Otherwise, recurse into children
    has_children = False
    for child in module.children():
        has_children = True
        reset_module_parameters(child)

    # For leaf modules without reset_parameters, reinitialize directly
    if not has_children:
        for name, param in module.named_parameters(recurse=False):
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                torch.nn.init.zeros_(param)


def randomize_encoder_block(model, block_idx):
    """Randomize all parameters in a specific encoder block."""
    block = model.vit.encoder.layer[block_idx]
    reset_module_parameters(block)


def randomize_embeddings(model):
    """Randomize patch embeddings and position embeddings."""
    # Patch embedding projection (Conv2d)
    reset_module_parameters(model.vit.embeddings.patch_embeddings)

    # Position embeddings (nn.Parameter, not a module)
    pos_emb = model.vit.embeddings.position_embeddings
    torch.nn.init.xavier_uniform_(pos_emb.data.unsqueeze(0))  # unsqueeze for 2D init
    # Squeeze back if needed — xavier works on 2D+ tensors
    # pos_emb is shape (1, num_patches+1, hidden_size), so we can init directly:
    torch.nn.init.normal_(pos_emb.data, mean=0.0, std=0.02)

    # CLS token
    cls_token = model.vit.embeddings.cls_token
    torch.nn.init.normal_(cls_token.data, mean=0.0, std=0.02)


def create_cascade_model(base_model, blocks_to_randomize, do_randomize_embeddings):
    """
    Create a cascaded-randomization copy of the model.
    The classification head is NEVER randomized.
    """
    model = copy.deepcopy(base_model)
    model.eval()

    for block_idx in blocks_to_randomize:
        randomize_encoder_block(model, block_idx)

    if do_randomize_embeddings:
        randomize_embeddings(model)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Create Adebayo cascading randomization models for ViT"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to fine-tuned ViT model (saved via Trainer/save_pretrained)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Parent directory for saving cascade models"
    )
    args = parser.parse_args()

    # ── Load base model ─────────────────────────────────────────────────
    print(f"Loading fine-tuned model from: {args.model_path}")
    base_model = ViTForImageClassification.from_pretrained(args.model_path)
    base_model.eval()

    # Also copy the processor so each level is self-contained
    processor = ViTImageProcessor.from_pretrained(args.model_path)

    # Verify architecture
    num_blocks = len(base_model.vit.encoder.layer)
    print(f"Model has {num_blocks} encoder blocks")
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Create cascade levels ───────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    for level_name, description, blocks, do_embeddings in CASCADE_LEVELS:
        level_path = os.path.join(args.output_dir, level_name)
        print(f"\n{'='*60}")
        print(f"Creating: {level_name}")
        print(f"  {description}")

        if blocks:
            print(f"  Randomizing encoder blocks: {blocks}")
        if do_embeddings:
            print(f"  Randomizing patch/position embeddings and CLS token")

        model = create_cascade_model(base_model, blocks, do_embeddings)

        # ── Quick sanity check: count how many params differ ────────
        n_changed = 0
        n_total = 0
        for (name_orig, p_orig), (name_new, p_new) in zip(
            base_model.named_parameters(), model.named_parameters()
        ):
            n_total += p_orig.numel()
            n_changed += (p_orig != p_new).sum().item()

        pct_changed = 100.0 * n_changed / n_total
        print(f"  Parameters changed: {n_changed:,} / {n_total:,} ({pct_changed:.1f}%)")

        # ── Save ────────────────────────────────────────────────────
        model.save_pretrained(level_path)
        processor.save_pretrained(level_path)
        print(f"  Saved to: {level_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary of cascade levels:")
    print(f"{'='*60}")
    for level_name, description, blocks, do_embeddings in CASCADE_LEVELS:
        level_path = os.path.join(args.output_dir, level_name)
        print(f"  {level_name:30s} -> {level_path}")
    print(f"\nAll models loadable via:")
    print(f"  ViTForImageClassification.from_pretrained('<level_path>')")


if __name__ == "__main__":
    main()
