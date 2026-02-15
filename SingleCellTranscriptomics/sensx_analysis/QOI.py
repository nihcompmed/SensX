import sys

sys.path.append('../model')
import model as ml


    # 4. Load Model
    model_path = exp_cfg['model_path']
    print(f"Loading model from {model_path}...")

    # Initialize model structure
    model = ml.BinaryClassifier(num_genes).to(device)

    # Load weights (weights_only=False required for your checkpoint format)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()





