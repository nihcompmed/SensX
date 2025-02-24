import numpy as np
import pickle
import model_library as ml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jrnd
from flax import nnx

from PIL import Image
from torchvision.transforms import v2 as T


def load_model(dataset):

    if dataset == 'Smiling':
        model_name = 'model_smiling.p'
    elif dataset == 'Eyeglasses':
        model_name = 'model_eyeglasses.p'

    
    dbfile = open(model_name, 'rb')
    model_state = pickle.load(dbfile)
    dbfile.close()
    
    model = ml.VisionTransformer(num_classes=1000)
    
    n_classes = 2
    
    model.classifier = nnx.Linear(model.classifier.in_features, n_classes, rngs=nnx.Rngs(0))
    
    nnx.update(model, model_state)


    return model


def QOI(x, model):

    model.eval()

    logits = model(x)

    softmax = jax.nn.softmax(logits, axis=1)

    return softmax[:,1]


def load_training_hypercube(dataset):

    if dataset == 'Smiling':
        hypercube_file = 'categorySmiling_timestamp1737675700_hypercube.npz'
    elif dataset == 'Eyeglasses':
        hypercube_file = 'categoryEyeglasses_timestamp1737861746_hypercube.npz'

    hypercube = np.load(hypercube_file)

    global_hypercube_low = hypercube['low']
    global_hypercube_high = hypercube['high']

    return global_hypercube_low, global_hypercube_high


def load_shortlisted_data_similing_eyeglasses():

    fname = 'shortlist_images_smiling_eyeglasses.p'

    dbfile = open(fname, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()

    XX = []

    for img in data:
        img_path = f'../../ViT_Jax/CelebA/img_align_celeba/{img}'
        image = Image.open(img_path)
        image = ml.tv_test_transforms(image)
        XX.append(image)

    XX = np.array(XX)

    return XX


def load_images_numpy(data):

    XX = []

    for img in data:
        img_path = f'../../ViT_Jax/CelebA/img_align_celeba/{img}'
        image = Image.open(img_path)
        image = ml.tv_test_transforms(image)

        XX.append(image)

    XX = np.array(XX)

    return XX









