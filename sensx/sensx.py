import torch
import numpy as np
from tqdm import tqdm

class SensitivityAnalyzer:
    def __init__(self, qoi_func, global_lower=0, global_upper=255, group_indices=None, device=None):
        """
        Modular Morris Method Sensitivity Analysis Engine (PyTorch).

        Three-step pipeline:
            1. compute_stability_profile — scan delta values, collect QoI statistics
            2. find_optimal_delta — select largest stable delta per (input, output)
            3. compute_sensitivity — compute elementary effects at chosen delta

        Args:
            qoi_func: Callable, (batch, *input_shape) -> (batch,) or (batch, M_total).
                Model or quantity-of-interest function. Must support batched input
                and return either a 1D scalar output or 2D multi-output tensor.
            global_lower: scalar or array-like, default 0.
                Lower bound(s) of the input domain. Scalar applies uniformly;
                array-like must broadcast to (D,) for per-feature bounds.
            global_upper: scalar or array-like, default 255.
                Upper bound(s) of the input domain. Same broadcasting as global_lower.
            group_indices: list of 1D LongTensors or None, default None.
                Each tensor contains feature indices belonging to one group.
                Groups must be disjoint (validated at init). When provided,
                sensitivity is computed per group rather than per feature.
            device: torch.device or None, default None.
                Device for all tensors. Defaults to CPU.
        """
        self.qoi_func = qoi_func
        self.device = device if device else torch.device("cpu")
        
        # --- Bounds Setup ---
        if np.isscalar(global_lower):
            self.global_lower = torch.tensor(global_lower, device=self.device, dtype=torch.float32)
        else:
            self.global_lower = torch.as_tensor(global_lower, device=self.device, dtype=torch.float32)
            
        if np.isscalar(global_upper):
            self.global_upper = torch.tensor(global_upper, device=self.device, dtype=torch.float32)
        else:
            self.global_upper = torch.as_tensor(global_upper, device=self.device, dtype=torch.float32)
            
        self.global_range = self.global_upper - self.global_lower

        # --- Grouping Setup ---
        self.group_indices = group_indices
        self.num_groups = len(group_indices) if group_indices else 0
        if self.num_groups > 0:
            # Ensure groups are on device
            self.group_indices = [g.to(self.device) for g in self.group_indices]
            # Validate no overlapping groups
            all_idx = torch.cat(self.group_indices)
            if all_idx.numel() != torch.unique(all_idx).numel():
                raise ValueError("Overlapping group_indices detected. Groups must be disjoint.")
            # Detect uniform group sizes for vectorized fast path
            group_sizes = [len(g) for g in self.group_indices]
            if len(set(group_sizes)) == 1:
                self.group_indices_stacked = torch.stack(self.group_indices)  # (num_groups, G)
                self.uniform_group_size = group_sizes[0]
            else:
                self.group_indices_stacked = None
                self.uniform_group_size = None
        else:
            self.group_indices_stacked = None
            self.uniform_group_size = None


    def compute_stability_profile(self, x, deltas, n_s, eval_batch_size, save_path=None):
        """
        Step 1: Scan delta values and collect QoI output statistics.

        For each delta, samples n_s random perturbations within the local
        neighborhood [x - delta*range, x + delta*range] (clamped to global
        bounds), evaluates qoi_func, and records quantiles and median.

        Args:
            x: Tensor, shape (N, *input_shape).
                Input samples. Flattened internally to (N, D).
            deltas: iterable of float.
                Candidate delta values to scan (fractions of global_range).
                Stored as-is in the output; need not be sorted.
            n_s: int.
                Number of random samples per delta.
            eval_batch_size: int.
                Batch size for qoi_func calls within each delta.
            save_path: str or None, default None.
                If provided, saves the profile as a .npz file at this path.
                The file can be loaded with np.load() and passed directly
                to find_optimal_delta.

        Returns:
            dict with keys:
                "deltas": np.ndarray, shape (num_deltas,), dtype float64.
                    Delta values in ascending order.
                "median": np.ndarray, shape (num_deltas, N) or (num_deltas, N, M).
                "q1":     np.ndarray, shape (num_deltas, N) or (num_deltas, N, M).
                "q99":    np.ndarray, shape (num_deltas, N) or (num_deltas, N, M).
                All statistic arrays are ordered along axis 0 to match "deltas".
        """
        x = x.to(dtype=torch.float32, device=self.device)
        N, D = x.shape[0], x[0].numel()
        original_shape = x.shape[1:]
        x_flat = x.reshape(N, D)

        oob = (x_flat < self.global_lower) | (x_flat > self.global_upper)
        if oob.any():
            n_feat = oob.any(dim=0).sum().item()
            n_samp = oob.any(dim=1).sum().item()
            print(f"  [Warning] {n_samp} input(s) have values outside global bounds across {n_feat} feature(s). "
                  f"Affected features will have collapsed perturbation ranges.")

        deltas_sorted = np.sort(np.asarray(deltas, dtype=np.float64))
        q_probs = torch.tensor([0.01, 0.99], device=self.device)

        median_list = []
        q1_list = []
        q99_list = []

        for delta in tqdm(deltas_sorted, desc="  Scanning Deltas"):
            delta_range = delta * self.global_range
            local_lower = torch.max(self.global_lower, x_flat - delta_range)
            local_upper = torch.min(self.global_upper, x_flat + delta_range)
            local_range = local_upper - local_lower

            qoi_accumulator = []
            num_batches = int(np.ceil(n_s / eval_batch_size))

            for b in range(num_batches):
                current_batch_size = min(eval_batch_size, n_s - b * eval_batch_size)
                rand_noise = torch.rand((current_batch_size, N, D), dtype=torch.float32, device=self.device)

                batch_samples_flat = local_lower.unsqueeze(0) + rand_noise * local_range.unsqueeze(0)
                model_input = batch_samples_flat.view(-1, *original_shape)

                with torch.no_grad():
                    raw_out = self.qoi_func(model_input)

                if raw_out.ndim > 1:
                     M = raw_out.shape[1]
                     batch_qois = raw_out.view(current_batch_size, N, M)
                else:
                     batch_qois = raw_out.view(current_batch_size, N)

                qoi_accumulator.append(batch_qois)

            all_qois = torch.cat(qoi_accumulator, dim=0)
            quantiles = torch.quantile(all_qois, q_probs, dim=0)
            median_val = torch.median(all_qois, dim=0).values

            median_list.append(median_val.cpu().numpy())
            q1_list.append(quantiles[0].cpu().numpy())
            q99_list.append(quantiles[1].cpu().numpy())

        profile = {
            "deltas": deltas_sorted,
            "median": np.stack(median_list, axis=0),
            "q1":     np.stack(q1_list, axis=0),
            "q99":    np.stack(q99_list, axis=0),
        }

        if save_path is not None:
            np.savez(save_path, **profile)
            print(f"  [Saved] Stability profile -> {save_path}")

        return profile

    def compute_sensitivity(self, x, delta_star, n_w, batch_size, target_output_indices, precision='float64'):
        """
        Step 3: Compute Morris method elementary effects.

        For each input, performs n_w random walks through the feature space.
        Each walk permutes features (or groups), perturbs them sequentially,
        and measures the absolute change in QoI output per unit perturbation.
        Results are averaged over all walks.

        Args:
            x: Tensor, shape (N, *input_shape).
                Input samples. Flattened internally to (N, D).
            delta_star: scalar, float, or Tensor of shape (N,).
                Perturbation radius as fraction of global_range.
                Scalar applies same delta to all inputs.
                Per-input vector allows different deltas per sample.
                Inputs with delta_star <= 0 are skipped (sensitivity = 0).
            n_w: int.
                Number of random walks per input.
            batch_size: int.
                Number of features (or groups) perturbed per forward pass.
                Controls the tradeoff between memory and kernel launches.
            target_output_indices: list, np.ndarray, or Tensor of int.
                Indices into qoi_func output to analyze. Length M.
                All M outputs share the same delta per input.
            precision: str, 'float32' or 'float64', default 'float64'.
                Controls dtype of trajectory state and accumulators.
                'float64': Prevents cumulative drift over long walks (up to D
                    sequential additions). Doubles memory for trajectory buffers.
                    qoi_func receives float32 via cast-on-call.
                'float32': Lower memory, faster on some GPUs. Acceptable when
                    D is small or drift tolerance is high.

        Returns:
            Tensor, shape (N, M, *input_shape), dtype float32.
                Mean absolute elementary effect per input, per output,
                per feature (in original spatial layout).
        """
        if precision not in ('float32', 'float64'):
            raise ValueError(f"precision must be 'float32' or 'float64', got '{precision}'")
        compute_dtype = torch.float64 if precision == 'float64' else torch.float32
        needs_cast = (compute_dtype == torch.float64)

        x = x.to(dtype=torch.float32, device=self.device)
        N = x.shape[0]
        original_shape = x.shape[1:]
        D = x[0].numel()
        x_flat = x.reshape(N, D)

        oob = (x_flat < self.global_lower) | (x_flat > self.global_upper)
        if oob.any():
            n_feat = oob.any(dim=0).sum().item()
            n_samp = oob.any(dim=1).sum().item()
            print(f"  [Warning] {n_samp} input(s) have values outside global bounds across {n_feat} feature(s). "
                  f"Affected features will have collapsed perturbation ranges.")

        if isinstance(target_output_indices, (np.ndarray, torch.Tensor)):
            target_output_indices = target_output_indices.tolist()
        M = len(target_output_indices)

        if not torch.is_tensor(delta_star):
            delta_star = torch.tensor(delta_star, device=self.device, dtype=torch.float32)
        else:
            delta_star = delta_star.to(dtype=torch.float32, device=self.device)

        # -------------------------------------------------------
        # DELTA NORMALIZATION
        # -------------------------------------------------------
        if delta_star.ndim == 0:
            delta_input = delta_star.expand(N)
        elif delta_star.ndim == 1 and delta_star.numel() == N:
            delta_input = delta_star
        else:
            raise ValueError(f"delta_star must be scalar or shape ({N},), got shape {delta_star.shape}")

        # -------------------------------------------------------
        # FILTERING INVALID DELTAS (delta <= 0)
        # -------------------------------------------------------
        valid_mask = (delta_input > 0)

        if not valid_mask.any():
            print("  [Engine Warning] No stable deltas found for any input! Returning zeros.")
            return torch.zeros((N, M, *original_shape), device=self.device)

        ignored_indices = torch.nonzero(~valid_mask).flatten().cpu().numpy()
        if len(ignored_indices) > 0:
            print(f"  [Engine Report] Ignoring {len(ignored_indices)} input(s) where delta_star == 0:")
            for idx in ignored_indices:
                print(f"    - Sample {idx}")

        valid_indices = torch.nonzero(valid_mask).flatten()
        x_active = x_flat[valid_indices]
        d_active = delta_input[valid_indices]

        # -------------------------------------------------------
        # CORE EXECUTION LOOP
        # -------------------------------------------------------
        results_list = []
        num_active = len(valid_indices)
        num_units = self.num_groups if self.group_indices else D
        g_range = self.global_range
        num_chunks = int(np.ceil(num_units / batch_size))

        # Pre-allocate reusable buffer. float64 prevents cumulative drift
        # across sequential additions; float32 saves memory.
        updates_buf = torch.zeros((batch_size, 1, D), dtype=compute_dtype, device=self.device)
        arange_full = torch.arange(batch_size, device=self.device)

        if self.group_indices_stacked is not None:
            row_idx_full = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, self.uniform_group_size)

        for i in tqdm(range(num_active), desc=f"  Computing ({num_active}/{N})"):
            x_curr = x_active[i : i+1]
            d_curr = d_active[i]

            feature_sens = torch.zeros((D, M), device=self.device, dtype=compute_dtype)

            local_lower = torch.max(self.global_lower, x_curr - (d_curr * g_range))
            local_upper = torch.min(self.global_upper, x_curr + (d_curr * g_range))
            local_range = local_upper - local_lower

            for w in tqdm(range(n_w)):
                z = local_lower + torch.rand_like(x_curr) * local_range
                diff = (z - x_curr).to(compute_dtype)
                permuted_order = torch.randperm(num_units, device=self.device)

                curr_input = x_curr.clone().to(compute_dtype).unsqueeze(0)  # (1, 1, D)
                with torch.no_grad():
                    qoi_input = curr_input.view(1, *original_shape)
                    if needs_cast:
                        qoi_input = qoi_input.float()
                    out_full = self.qoi_func(qoi_input)
                    last_qoi = out_full.view(-1)[target_output_indices].view(1, M).to(compute_dtype)

                for k in tqdm(range(num_chunks)):
                    chunk_indices = permuted_order[k*batch_size : (k+1)*batch_size]
                    actual_bs = len(chunk_indices)
                    updates = updates_buf[:actual_bs]
                    arange_bs = arange_full[:actual_bs]

                    if self.group_indices:
                        if self.group_indices_stacked is not None:
                            chunk_group_idx = self.group_indices_stacked[chunk_indices]
                            row_idx = row_idx_full[:actual_bs]
                            updates[row_idx, 0, chunk_group_idx] = diff[0, chunk_group_idx]
                            dx_groups = torch.norm(diff[0, chunk_group_idx].to(compute_dtype), p=2, dim=1)
                        else:
                            dx_groups = torch.zeros(actual_bs, device=self.device, dtype=compute_dtype)
                            for row, unit_idx in enumerate(chunk_indices):
                                idx_mask = self.group_indices[unit_idx]
                                updates[row, 0, idx_mask] = diff[0, idx_mask]
                                dx_groups[row] = torch.norm(diff[0, idx_mask], p=2).to(compute_dtype)
                    else:
                        updates[arange_bs, 0, chunk_indices] = diff[0, chunk_indices]
                        dx_groups = torch.abs(diff[0, chunk_indices]).to(compute_dtype)

                    batch_inputs = curr_input + torch.cumsum(updates, dim=0)

                    # Selective clear: zero only the entries we touched
                    if self.group_indices:
                        if self.group_indices_stacked is not None:
                            updates[row_idx, 0, chunk_group_idx] = 0
                        else:
                            for row, unit_idx in enumerate(chunk_indices):
                                updates[row, 0, self.group_indices[unit_idx]] = 0
                    else:
                        updates[arange_bs, 0, chunk_indices] = 0

                    with torch.no_grad():
                        qoi_input = batch_inputs.view(actual_bs, *original_shape)
                        if needs_cast:
                            qoi_input = qoi_input.float()
                        out_batch_full = self.qoi_func(qoi_input)
                        out_batch = out_batch_full.view(actual_bs, -1)[:, target_output_indices].to(compute_dtype)

                    prev_qois = torch.cat([last_qoi, out_batch[:-1]], dim=0)
                    dy = out_batch - prev_qois

                    dx_view = dx_groups.view(actual_bs, 1)
                    valid_mask_ee = (dx_groups > 1e-12)
                    ee = torch.zeros_like(dy)
                    ee[valid_mask_ee] = torch.abs(dy[valid_mask_ee] / dx_view[valid_mask_ee])

                    if self.group_indices:
                        if self.group_indices_stacked is not None:
                            ee_expanded = ee.unsqueeze(1).expand(-1, self.uniform_group_size, -1)
                            feat_idx_flat = chunk_group_idx.reshape(-1)
                            ee_flat = ee_expanded.reshape(-1, ee.shape[1])
                            feature_sens.index_add_(0, feat_idx_flat, ee_flat)
                        else:
                            for row, unit_idx in enumerate(chunk_indices):
                                feature_sens[self.group_indices[unit_idx]] += ee[row].unsqueeze(0)
                    else:
                        feature_sens[chunk_indices] += ee

                    last_qoi = out_batch[-1:]
                    curr_input = batch_inputs[-1:]

            results_list.append((feature_sens / n_w).t().unsqueeze(0))

        # -------------------------------------------------------
        # RECONSTRUCTION
        # -------------------------------------------------------
        computed_tensor = torch.cat(results_list, dim=0)  # (num_active, M, D)
        final_tensor = torch.zeros((N, M, D), device=self.device, dtype=torch.float32)
        final_tensor[valid_indices] = computed_tensor.float()
        return final_tensor.view(N, M, *original_shape)




def find_optimal_delta(stability_profile, tau_a):
    """
    Step 2: Select the largest stable delta per (input, output) pair.

    For each delta (in ascending order), checks whether all larger deltas
    produce QoI variation (q99 - q1) within tolerance tau_a. Selects the
    smallest delta where this holds — i.e., the largest perturbation range
    that keeps the QoI stable beyond that point.

    Note: The last delta in the profile is never selected as a candidate
    because there are no larger deltas to verify stability against.

    Args:
        stability_profile: dict with keys "deltas", "q1", "q99", "median".
            Output of compute_stability_profile.
            "deltas": np.ndarray, shape (num_deltas,), ascending order.
            "q1":     np.ndarray, shape (num_deltas, N) or (num_deltas, N, M).
            "q99":    np.ndarray, shape (num_deltas, N) or (num_deltas, N, M).
        tau_a: scalar or array-like of shape (M,).
            Stability tolerance per output. Scalar applies to all outputs.

    Returns:
        Tensor, shape (N, M), dtype float32.
            Optimal delta per (input, output):
            - delta in (0, 1): stable plateau found at this delta.
            - 1.0: sensitive at all scanned scales (never stabilizes),
              uses full domain perturbation.
            - 0.0: insensitive (output variation within tau_a across the
              entire scanned range) — skipped by compute_sensitivity.
    """
    deltas = stability_profile["deltas"]
    q1_stack = stability_profile["q1"]
    q99_stack = stability_profile["q99"]

    if q1_stack.ndim == 2:
        q1_stack = q1_stack[..., np.newaxis]
        q99_stack = q99_stack[..., np.newaxis]

    N = q1_stack.shape[1]
    M = q1_stack.shape[2]

    if np.isscalar(tau_a):
        tau_a = np.full(M, tau_a)
    else:
        tau_a = np.array(tau_a)

    # Initialize with 0.0 to detect failures later
    optimal_deltas = np.zeros((N, M), dtype=np.float32)
    valid_matrix = np.zeros((len(deltas), N, M), dtype=bool)

    # Check if output is insensitive across the entire scanned range
    full_variation = np.max(q99_stack, axis=0) - np.min(q1_stack, axis=0)
    insensitive_mask = (full_variation <= tau_a[np.newaxis, :])

    for j in range(len(deltas) - 1):
        variation = np.max(q99_stack[j+1:, ...], axis=0) - np.min(q1_stack[j+1:, ...], axis=0)
        valid_matrix[j, ...] = (variation <= tau_a[np.newaxis, :])

    has_valid = np.any(valid_matrix, axis=0) & ~insensitive_mask  # (N, M)
    first_valid_idx = np.argmax(valid_matrix, axis=0)             # (N, M)
    optimal_deltas[has_valid] = deltas[first_valid_idx[has_valid]]

    # Never stabilizes and not insensitive → sensitive at all scales → full domain
    always_sensitive = ~np.any(valid_matrix, axis=0) & ~insensitive_mask  # (N, M)
    optimal_deltas[always_sensitive] = 1.0

    # Returns 0.0 only where the input is insensitive to this output
    return optimal_deltas

