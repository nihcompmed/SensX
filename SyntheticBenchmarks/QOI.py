import functools
import torch
import model as ml


def qoi_wrapper(model_path, input_dim, device):

    model = ml.FeedForwardNet(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model.predict_proba


