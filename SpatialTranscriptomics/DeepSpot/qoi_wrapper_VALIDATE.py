import torch
import torch.nn.functional as F
import numpy as np
from functools import partial

def gpu_preprocess(tensor, mean, std):
    if tensor.dim() == 5:
        N, M, H, W, C = tensor.shape
        t = tensor.reshape(N * M, H, W, C)
    else:
        t = tensor 
    
    t = t.permute(0, 3, 1, 2).float() * (1.0 / 255.0)
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
    t = (t - mean) / std
    
    if tensor.dim() == 5:
        return t.view(N, M, 3, 224, 224)
    else:
        return t

def qoi_function(xx, mean, std, resize_shape, model_expression, morphology_model, device, output_idxs, micro_batch_size=1024):
    with torch.no_grad():
        if isinstance(xx, np.ndarray):
            xx = torch.from_numpy(xx)
        
        if xx.device != device:
            xx = xx.to(device)

        B = xx.shape[0]

        # Critical fix: added .contiguous() before view
        all_X = xx.contiguous().view(B, 9, 100, 100, 3)

        X_spots = all_X[:, 0]      
        X_neighbors = all_X[:, 1:] 

        tiles = X_spots[:, :99, :99, :].unfold(1, 33, 33).unfold(2, 33, 33)
        tiles = tiles.permute(0, 1, 2, 4, 5, 3).contiguous()
        X_subspots = tiles.view(B, 9, 33, 33, 3)
        
        p_spots = gpu_preprocess(X_spots, mean, std)         
        p_subspots = gpu_preprocess(X_subspots, mean, std)   
        p_neighbors = gpu_preprocess(X_neighbors, mean, std) 

        mega_batch = torch.cat([
            p_spots.unsqueeze(1), 
            p_subspots, 
            p_neighbors
        ], dim=1).view(-1, 3, 224, 224)

        all_features_list = []
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i in range(0, mega_batch.shape[0], micro_batch_size):
                chunk = mega_batch[i : i + micro_batch_size]
                all_features_list.append(morphology_model(chunk))
            
            all_features = torch.cat(all_features_list, dim=0)
            all_features = all_features.view(B, 18, -1).to(torch.float32)

        f_spot = all_features[:, 0:1].contiguous()    
        f_sub = all_features[:, 1:10].contiguous()   
        f_neigh = all_features[:, 10:18].contiguous() 

        preds = model_expression([f_spot, f_sub, f_neigh]) 
        
        return preds[:, output_idxs].to(torch.float32)

def qoi_wrapper(resize_shape, model_expression, preprocess, compute_mini_tiles, detach_and_convert, morphology_model, mean, std, device, output_idxs, max_batch_size=1000):
    print("Compiling morphology_model...")
    morphology_model = torch.compile(morphology_model)
    
    print(f"Running End-to-End Warmup...")
    flat_dim = 9 * 100 * 100 * 3
    dummy_xx = np.zeros((max_batch_size, flat_dim), dtype=np.uint8)
    
    _ = qoi_function(dummy_xx, mean, std, resize_shape, model_expression, morphology_model, device, output_idxs)
    torch.cuda.synchronize()
    print("Warm-up complete.")

    return partial(qoi_function, 
                   mean=mean,
                   std=std,
                   resize_shape=resize_shape, 
                   model_expression=model_expression, 
                   morphology_model=morphology_model, 
                   device=device,
                   output_idxs=output_idxs)
