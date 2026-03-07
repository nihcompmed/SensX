import sys
import functools
import torch

sys.path.append('../model')
import model_library as ml

def qoi_wrapper(model_path, num_genes, device):


    # Initialize model structure
    model = ml.BinaryClassifier(num_genes).to(device)

    # Load weights (weights_only=False required for your checkpoint format)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    qoi = functools.partial(ml.probability_qoi, model=model)

    return qoi





