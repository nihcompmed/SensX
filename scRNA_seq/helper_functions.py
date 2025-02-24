import numpy as np
import pickle
import model_library as ml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jrnd
from flax import nnx


def load_model(model_name):
    
    # load model
    model_layers = [56239,250,50,2]
    
    dbfile = open(model_name, 'rb')
    model_state = pickle.load(dbfile)
    dbfile.close()
    
    model = ml.Model(model_layers, rngs=nnx.Rngs(0))
    
    nnx.update(model, model_state)

    return model

def QOI(x, model):

    model.eval()

    logits = model(x)

    softmax = jax.nn.softmax(logits, axis=1)

    return softmax[:,1]


def load_test_data(ctype_class):

    fname = f'test_shortlisted/shortlisted_test_class{ctype_class}.npy'

    return np.load(fname)









