import torch
import numpy as np
from tqdm import tqdm

class SensitivityAnalyzer:
    def __init__(self, qoi_func, global_lower=0, global_upper=255, group_indices=None, device=None):
        """
        Generic Sensitivity Analysis Engine (PyTorch) with Grouping Support.
        Auto-detects Scalar/Vector/Matrix delta strategies.
        Skips computation for any instance where delta_star <= 0.
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

    def compute_stability_profile(self, x, deltas, n_s, eval_batch_size):
        # ... (Same as before) ...
        x = x.to(dtype=torch.float32, device=self.device)
        N, D = x.shape[0], x[0].numel()
        original_shape = x.shape[1:] 
        x_flat = x.reshape(N, D)
        profile = {}
        
        q_probs = torch.tensor([0.01, 0.99], device=self.device)

        for delta in tqdm(deltas, desc="  Scanning Deltas"):
            delta_range = delta * self.global_range 
            local_lower = torch.max(self.global_lower, x_flat - delta_range)
            local_upper = torch.min(self.global_upper, x_flat + delta_range)
            
            qoi_accumulator = []
            num_batches = int(np.ceil(n_s / eval_batch_size))
            
            for b in range(num_batches):
                current_batch_size = min(eval_batch_size, n_s - b * eval_batch_size)
                rand_noise = torch.rand((current_batch_size, N, D), dtype=torch.float32, device=self.device)
                
                batch_samples_flat = local_lower.unsqueeze(0) + rand_noise * (local_upper.unsqueeze(0) - local_lower.unsqueeze(0))
                model_input = batch_samples_flat.view(-1, *original_shape)
                
                with torch.no_grad():
                    raw_out = self.qoi_func(model_input)
                
                if raw_out.ndim > 1 and raw_out.shape[1] > 1:
                     M = raw_out.shape[1]
                     batch_qois = raw_out.view(current_batch_size, N, M)
                else:
                     batch_qois = raw_out.view(current_batch_size, N)
                
                qoi_accumulator.append(batch_qois)

            all_qois = torch.cat(qoi_accumulator, dim=0)
            quantiles = torch.quantile(all_qois, q_probs, dim=0) 
            median_val = torch.median(all_qois, dim=0).values
            
            profile[delta] = {
                "median": median_val.cpu().numpy(), 
                "q1": quantiles[0].cpu().numpy(), 
                "q99": quantiles[1].cpu().numpy()
            }
        return profile

    def find_optimal_delta(self, stability_profile, tau_a):
        # ... (Same as before) ...
        sorted_deltas = sorted(stability_profile.keys())
        deltas = np.array(sorted_deltas)
        
        q1_stack = np.stack([stability_profile[d]['q1'] for d in sorted_deltas], axis=0)
        q99_stack = np.stack([stability_profile[d]['q99'] for d in sorted_deltas], axis=0)
        
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
        
        for j in range(len(deltas) - 1):
            variation = np.max(q99_stack[j+1:, ...], axis=0) - np.min(q1_stack[j+1:, ...], axis=0)
            valid_matrix[j, ...] = (variation <= tau_a[np.newaxis, :])

        for i in range(N):
            for m in range(M):
                if np.any(valid_matrix[:, i, m]):
                    idx = np.argmax(valid_matrix[:, i, m])
                    optimal_deltas[i, m] = deltas[idx]
        
        # Returns 0 where no stable delta was found
        return torch.from_numpy(optimal_deltas).to(dtype=torch.float32, device=self.device)

    def compute_sensitivity(self, x, delta_star, n_w, batch_size, target_output_indices):
        """
        Unified Sensitivity Engine handling 3 Cases.
        Skips any (input, output) pair where delta_star == 0.
        """
        x = x.to(dtype=torch.float32, device=self.device)
        N, D = x.shape
        
        if isinstance(target_output_indices, (np.ndarray, torch.Tensor)):
            target_output_indices = target_output_indices.tolist()
        M = len(target_output_indices)

        if not torch.is_tensor(delta_star):
            delta_star = torch.tensor(delta_star, device=self.device, dtype=torch.float32)
        else:
            delta_star = delta_star.to(dtype=torch.float32, device=self.device)

        # -------------------------------------------------------
        # CASE DETECTION & EXPANSION
        # -------------------------------------------------------
        # The goal is to standardize everything to the expanded "List of Tasks" format
        # Task = (Input Vector, Delta Scalar, Target Indices List)
        
        # --- CASE 1: Scalar ---
        if delta_star.ndim == 0:
            x_input = x
            delta_input = delta_star.view(1).repeat(N)
            target_map = [target_output_indices] * N
            expanded_mode = False
            
        # --- CASE 2: Vector (N,) ---
        elif delta_star.ndim == 1 and delta_star.numel() == N:
            x_input = x
            delta_input = delta_star
            target_map = [target_output_indices] * N
            expanded_mode = False

        # --- CASE 3: Matrix (N, M) ---
        elif delta_star.ndim == 2 and delta_star.shape == (N, M):
            x_input = x.repeat_interleave(M, dim=0) # (N*M, D)
            delta_input = delta_star.view(-1)       # (N*M,)
            raw_targets = target_output_indices * N
            target_map = [[t] for t in raw_targets] 
            expanded_mode = True
            
        else:
            raise ValueError(f"Invalid delta_star shape {delta_star.shape}")

        # -------------------------------------------------------
        # FILTERING INVALID DELTAS (delta <= 0)
        # -------------------------------------------------------
        valid_mask = (delta_input > 0)
        num_total_tasks = len(delta_input)
        
        if not valid_mask.any():
            print("  [Engine Warning] No stable deltas found for any input! Returning zeros.")
            # Return shape depends on mode
            if expanded_mode: return torch.zeros((N, M, D), device=self.device)
            else: return torch.zeros((N, M, D), device=self.device)

        # Report Ignored Tuples
        ignored_indices = torch.nonzero(~valid_mask).squeeze(1).cpu().numpy()
        if len(ignored_indices) > 0:
            print(f"  [Engine Report] Ignoring {len(ignored_indices)} cases where delta_star == 0:")
            for idx in ignored_indices:
                if expanded_mode:
                    # Map flat index back to (sample, output)
                    # idx = sample * M + output_relative_idx
                    s_idx = idx // M
                    o_rel = idx % M
                    o_real = target_output_indices[o_rel]
                    print(f"    - Sample {s_idx}, Output Index {o_real} (Relative {o_rel})")
                else:
                    # In Case 1/2, idx maps directly to Sample
                    print(f"    - Sample {idx} (All Targets)")

        # Subset Data for Computation
        valid_indices = torch.nonzero(valid_mask).squeeze(1)
        
        x_active = x_input[valid_indices]
        d_active = delta_input[valid_indices]
        
        # target_map is a list, need to subset via list comprehension
        # (Converting tensor idx to int for list access)
        t_active = [target_map[i.item()] for i in valid_indices]

        # -------------------------------------------------------
        # CORE EXECUTION LOOP (On Active Data Only)
        # -------------------------------------------------------
        results_list = []
        num_active = len(t_active)
        num_units = self.num_groups if self.group_indices else D
        g_range = self.global_range

        for i in tqdm(range(num_active), desc=f"  Computing ({num_active}/{num_total_tasks})"):
            print(f'  Computing ({num_active}/{num_total_tasks})')
            x_curr = x_active[i : i+1]
            d_curr = d_active[i]
            targets_curr = t_active[i]
            m_curr = len(targets_curr)
            
            # Accumulator
            feature_sens = torch.zeros((D, m_curr), device=self.device, dtype=torch.float64)
            
            local_lower = torch.max(self.global_lower, x_curr - (d_curr * g_range))
            local_upper = torch.min(self.global_upper, x_curr + (d_curr * g_range))

            for w in tqdm(range(n_w)):
                z = local_lower + torch.rand_like(x_curr) * (local_upper - local_lower)
                diff = z - x_curr
                permuted_order = torch.randperm(num_units, device=self.device)
                
                curr_input = x_curr.clone()
                with torch.no_grad():
                    out_full = self.qoi_func(curr_input)
                    last_qoi = out_full.view(-1)[targets_curr].view(1, m_curr).to(torch.float64)

                num_chunks = int(np.ceil(num_units / batch_size))
                for k in tqdm(range(num_chunks)):
                    chunk_indices = permuted_order[k*batch_size : (k+1)*batch_size]
                    actual_bs = len(chunk_indices)
                    
                    updates = torch.zeros((actual_bs, 1, D), dtype=torch.float32, device=self.device)
                    dx_groups = torch.zeros(actual_bs, device=self.device, dtype=torch.float64)

                    for row, unit_idx in enumerate(chunk_indices):
                        if self.group_indices:
                            idx_mask = self.group_indices[unit_idx]
                            updates[row, 0, idx_mask] = diff[0, idx_mask]
                            dx_groups[row] = torch.norm(diff[0, idx_mask], p=2).to(torch.float64)
                        else:
                            updates[row, 0, unit_idx] = diff[0, unit_idx]
                            dx_groups[row] = torch.abs(diff[0, unit_idx]).to(torch.float64)

                    batch_inputs = curr_input + torch.cumsum(updates, dim=0)

                    with torch.no_grad():
                        out_batch_full = self.qoi_func(batch_inputs)
                        out_batch = out_batch_full[:, targets_curr].view(actual_bs, m_curr).to(torch.float64)

                    prev_qois = torch.cat([last_qoi, out_batch[:-1]], dim=0)
                    dy = out_batch - prev_qois 
                    
                    dx_view = dx_groups.view(actual_bs, 1)
                    valid_mask_ee = (dx_groups > 1e-12)
                    ee = torch.zeros_like(dy)
                    ee[valid_mask_ee] = torch.abs(dy[valid_mask_ee] / dx_view[valid_mask_ee])

                    for row, unit_idx in enumerate(chunk_indices):
                        if self.group_indices:
                            feature_sens[self.group_indices[unit_idx]] += ee[row].unsqueeze(0)
                        else:
                            feature_sens[unit_idx] += ee[row]

                    last_qoi = out_batch[-1].unsqueeze(0)
                    curr_input = batch_inputs[-1].unsqueeze(0)

            results_list.append((feature_sens / n_w).t().unsqueeze(0))

        # -------------------------------------------------------
        # RECONSTRUCTION (Filling the holes)
        # -------------------------------------------------------
        computed_tensor = torch.cat(results_list, dim=0) # (Num_Active, m_curr, D)
        
        # We need to map these back to the full shape.
        # Logic differs slightly between Expanded (Case 3) and Standard (Case 1/2)
        
        if expanded_mode:
            # Full shape: (N*M, 1, D) -> Reshape to (N, M, D)
            # Init with Zeros
            final_tensor = torch.zeros((num_total_tasks, 1, D), device=self.device, dtype=torch.float32)
            # Fill active slots
            final_tensor[valid_indices] = computed_tensor.float()
            # Reshape
            return final_tensor.view(N, M, D)
            
        else:
            # Full shape: (N, M, D)
            final_tensor = torch.zeros((N, M, D), device=self.device, dtype=torch.float32)
            # Fill active slots
            final_tensor[valid_indices] = computed_tensor.float()
            return final_tensor

    def compute_sensitivity_wide(self, x, delta_star, n_w, input_batch_size, target_output_indices):
        """
        Input-Batched Sensitivity Engine with Grouping Support.
        Processes multiple inputs simultaneously per forward pass.
        Shares a single permuted order of units (features or groups) across all
        inputs in a batch.

        Handles the same 3 delta_star cases and zero-delta filtering as
        compute_sensitivity.

        WARNING: Iterates over units (features/groups) one at a time.
                 Only efficient for low-D data or grouped features.
        """
        x = x.to(dtype=torch.float32, device=self.device)
        N, D = x.shape

        if isinstance(target_output_indices, (np.ndarray, torch.Tensor)):
            target_output_indices = target_output_indices.tolist()
        M = len(target_output_indices)

        if not torch.is_tensor(delta_star):
            delta_star = torch.tensor(delta_star, device=self.device, dtype=torch.float32)
        else:
            delta_star = delta_star.to(dtype=torch.float32, device=self.device)

        # -------------------------------------------------------
        # CASE DETECTION & EXPANSION
        # -------------------------------------------------------
        # --- CASE 1: Scalar ---
        if delta_star.ndim == 0:
            x_input = x
            delta_input = delta_star.view(1).repeat(N)
            target_map = [target_output_indices] * N
            expanded_mode = False

        # --- CASE 2: Vector (N,) ---
        elif delta_star.ndim == 1 and delta_star.numel() == N:
            x_input = x
            delta_input = delta_star
            target_map = [target_output_indices] * N
            expanded_mode = False

        # --- CASE 3: Matrix (N, M) ---
        elif delta_star.ndim == 2 and delta_star.shape == (N, M):
            x_input = x.repeat_interleave(M, dim=0)  # (N*M, D)
            delta_input = delta_star.view(-1)          # (N*M,)
            raw_targets = target_output_indices * N
            target_map = [[t] for t in raw_targets]
            expanded_mode = True

        else:
            raise ValueError(f"Invalid delta_star shape {delta_star.shape}")

        # -------------------------------------------------------
        # FILTERING INVALID DELTAS (delta <= 0)
        # -------------------------------------------------------
        valid_mask = (delta_input > 0)
        num_total_tasks = len(delta_input)

        if not valid_mask.any():
            print("  [Engine Warning] No stable deltas found for any input! Returning zeros.")
            if expanded_mode:
                return torch.zeros((N, M, D), device=self.device)
            else:
                return torch.zeros((N, M, D), device=self.device)

        # Report Ignored Tuples
        ignored_indices = torch.nonzero(~valid_mask).squeeze(1).cpu().numpy()
        if len(ignored_indices) > 0:
            print(f"  [Engine Report] Ignoring {len(ignored_indices)} cases where delta_star == 0:")
            for idx in ignored_indices:
                if expanded_mode:
                    s_idx = idx // M
                    o_rel = idx % M
                    o_real = target_output_indices[o_rel]
                    print(f"    - Sample {s_idx}, Output Index {o_real} (Relative {o_rel})")
                else:
                    print(f"    - Sample {idx} (All Targets)")

        # Subset Data for Computation
        valid_indices = torch.nonzero(valid_mask).squeeze(1)

        x_active = x_input[valid_indices]
        d_active = delta_input[valid_indices]
        t_active = [target_map[i.item()] for i in valid_indices]

        # -------------------------------------------------------
        # CORE EXECUTION (Input-Batched, Unit-Sequential)
        # -------------------------------------------------------
        num_active = len(t_active)
        num_units = self.num_groups if self.group_indices else D
        g_range = self.global_range

        # All active tasks must share the same target list for batching to work.
        # In Case 1/2 this is guaranteed. In Case 3 (expanded), each task has
        # a single (potentially different) target, so we must sub-batch by target.
        # Group active tasks by their target list for batched processing.
        from collections import OrderedDict
        target_groups = OrderedDict()
        for flat_idx, targets in enumerate(t_active):
            key = tuple(targets)
            if key not in target_groups:
                target_groups[key] = []
            target_groups[key].append(flat_idx)

        # Result buffer: one (m_curr, D) per active task
        all_results = [None] * num_active

        for targets_tuple, task_indices in target_groups.items():
            targets_curr = list(targets_tuple)
            m_curr = len(targets_curr)
            task_indices_t = torch.tensor(task_indices, device=self.device, dtype=torch.long)

            x_group = x_active[task_indices_t]       # (G, D)
            d_group = d_active[task_indices_t]        # (G,)
            G = x_group.shape[0]

            # Process in input-batch chunks
            for chunk_start in range(0, G, input_batch_size):
                chunk_end = min(chunk_start + input_batch_size, G)
                chunk_bs = chunk_end - chunk_start

                x_chunk = x_group[chunk_start:chunk_end]            # (chunk_bs, D)
                d_chunk = d_group[chunk_start:chunk_end].view(-1, 1) # (chunk_bs, 1)

                # Accumulator in Float64: (chunk_bs, D, m_curr)
                feature_sens = torch.zeros((chunk_bs, D, m_curr), device=self.device, dtype=torch.float64)

                for w in tqdm(range(n_w), desc=f"    Batch {chunk_start}-{chunk_end} of {G}", leave=False):
                    delta_range = d_chunk * g_range.unsqueeze(0)     # (chunk_bs, D)
                    local_lower = torch.max(self.global_lower, x_chunk - delta_range)
                    local_upper = torch.min(self.global_upper, x_chunk + delta_range)

                    z_batch = local_lower + torch.rand_like(x_chunk) * (local_upper - local_lower)
                    diff_batch = z_batch - x_chunk                   # (chunk_bs, D)

                    permuted_order = torch.randperm(num_units, device=self.device)
                    current_x = x_chunk.clone()

                    with torch.no_grad():
                        out_full = self.qoi_func(current_x)
                        last_qois = out_full[:, targets_curr].view(chunk_bs, m_curr).to(torch.float64)

                    # Walk through units one at a time
                    for unit_idx in permuted_order:
                        if self.group_indices:
                            idx_mask = self.group_indices[unit_idx]
                            current_x[:, idx_mask] = z_batch[:, idx_mask]
                            dx = torch.norm(diff_batch[:, idx_mask], p=2, dim=1).to(torch.float64)  # (chunk_bs,)
                        else:
                            current_x[:, unit_idx] = z_batch[:, unit_idx]
                            dx = torch.abs(diff_batch[:, unit_idx]).to(torch.float64)                # (chunk_bs,)

                        with torch.no_grad():
                            out_new = self.qoi_func(current_x)
                            new_qois = out_new[:, targets_curr].view(chunk_bs, m_curr).to(torch.float64)

                        dy = new_qois - last_qois                    # (chunk_bs, m_curr)
                        dx_view = dx.view(chunk_bs, 1)               # (chunk_bs, 1)
                        valid_mask_ee = (dx > 1e-12)                 # (chunk_bs,)

                        ee = torch.zeros_like(dy)                    # (chunk_bs, m_curr)
                        ee[valid_mask_ee] = torch.abs(dy[valid_mask_ee] / dx_view[valid_mask_ee])

                        if self.group_indices:
                            idx_mask = self.group_indices[unit_idx]
                            # Broadcast ee to all features in the group
                            for feat in idx_mask:
                                feature_sens[:, feat, :] += ee
                        else:
                            feature_sens[:, unit_idx, :] += ee

                        last_qois = new_qois

                feature_sens /= n_w  # (chunk_bs, D, m_curr)

                # Store results: transpose to (chunk_bs, m_curr, D) to match compute_sensitivity output
                chunk_results = feature_sens.permute(0, 2, 1)  # (chunk_bs, m_curr, D)
                chunk_task_indices = task_indices[chunk_start:chunk_end]
                for local_i, global_i in enumerate(chunk_task_indices):
                    all_results[global_i] = chunk_results[local_i].unsqueeze(0)

        # -------------------------------------------------------
        # RECONSTRUCTION (Filling the holes)
        # -------------------------------------------------------
        computed_tensor = torch.cat(all_results, dim=0)  # (num_active, m_curr, D)

        if expanded_mode:
            final_tensor = torch.zeros((num_total_tasks, 1, D), device=self.device, dtype=torch.float32)
            final_tensor[valid_indices] = computed_tensor.float()
            return final_tensor.view(N, M, D)
        else:
            final_tensor = torch.zeros((N, M, D), device=self.device, dtype=torch.float32)
            final_tensor[valid_indices] = computed_tensor.float()
            return final_tensor
