import numpy as np
import pickle
import model_library as ml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jrnd
from flax import nnx


def load_model(dataset):

    # Load training log
    dbfile = open('training_log_testf1.p', 'rb')
    all_log_testf1 = pickle.load(dbfile)
    dbfile.close()
    
    
    # load model
    model_layers = all_log_testf1[dataset]['model_layers']
    model_name = all_log_testf1[dataset]['model_name']
    
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


def load_data(dataset, n_test):

    # Load training data
    
    dbfile = open('synthetic_data.p', 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    
    load_data = data[dataset]
    
    X = load_data['X']
    
    train_idxs = load_data['train_idxs']
    X_train = X[train_idxs]
    
    # Load test data (true positives)
    
    dbfile = open('synthetic_test_positive_data.p', 'rb')
    test_data_dict = pickle.load(dbfile)
    dbfile.close()
    
    # Batch
    X_test = test_data_dict[dataset]['samples']
    test_datatypes = test_data_dict[dataset]['datatypes']

    X_test = X_test[:n_test]
    if test_datatypes is not None:
        test_datatypes = test_datatypes[:n_test]

    if n_test == 1:
        X_test = np.expand_dims(X_test, axis=0)

    return X_train, X_test, test_datatypes



