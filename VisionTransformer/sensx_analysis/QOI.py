import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import torchvision.transforms as T
import os
import functools


def ViTQOI(x, model):

    # Inference
    outputs = model(x)
    
    # Extract Probability of Class 1 (Smiling)
    probs = torch.softmax(outputs.logits, dim=1)

    return probs[:, 1] 

# 2. Function to Initialize Model & QOI
def initialize_model_and_qoi(model_path, device):
    print(f"Loading ViT from {model_path}...")
    
    # Load Model
    model = ViTForImageClassification.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    
    # Create QOI Instance
    qoi_func = functools.partial(ViTQOI, model=model)

    # Load Processor (for mean/std stats and size)
    processor = ViTImageProcessor.from_pretrained(model_path)
    
    # Define Input Preprocessing (Raw Resize -> Tensor)
    # This transforms the raw PIL image to the 0-1 tensor the analysis expects
    height, width = processor.size['height'], processor.size['width']
    preprocessing_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor()
    ])
    
    return qoi_func, preprocessing_transform

