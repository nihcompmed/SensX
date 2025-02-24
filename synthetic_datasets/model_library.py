import jax
jax.config.update("jax_enable_x64", True)
from flax import nnx
import optax
import jax.numpy as jnp

import numpy as np
import math

from copy import deepcopy
import time
import pickle

class Model(nnx.Module):

    def __init__(self, layer_widths, rngs:nnx.Rngs):

        self.linear_layers = []

        self.n_layers = len(layer_widths)-1

        for j in range(self.n_layers):
            self.linear_layers.append(nnx.Linear(layer_widths[j], layer_widths[j+1], rngs=rngs))

    def __call__(self, x):

        for j in range(self.n_layers-1):
            x = nnx.relu(self.linear_layers[j](x))

        return self.linear_layers[-1](x)



@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, batch, labels):

  def loss_fn(model):

    logits = model(batch)  # call methods directly

    loss_value = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    loss_value = jnp.mean(loss_value)

    return loss_value

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # in-place updates

  return loss
    

# samples: all samples
# labels: all labels
def train_save(\
        samples\
        , labels\
        , train_idxs\
        , test_idxs\
        , model_layers=[10, 200, 100, 50, 2]
        , num_epochs=100\
        , num_train_batches=100\
        , training_batchsize=100\
        , learning_rate=0.0001\
        , rng_seed=0\
        , save_suffix=''\
        ):

    model = Model(model_layers, rngs=nnx.Rngs(rng_seed))

    # Initialize optimization
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate))  # reference sharing

    nn = len(train_idxs)

    assert training_batchsize < nn

    train_samples = samples[train_idxs]
    test_samples = samples[test_idxs]

    train_labels = labels[train_idxs]
    test_labels = labels[test_idxs]

    record_test_metric = []

    max_f1 = -math.inf

    for epoch in range(num_epochs):

        model.train()

        for ii in range(num_train_batches):


            batch_idxs = np.random.choice(nn, size=training_batchsize, replace=False)

            batch_samples = train_samples[batch_idxs]

            batch_labels = train_labels[batch_idxs]
            
            loss_value = train_step(model\
                            , optimizer\
                            , batch_samples\
                            , batch_labels)

                

        print(f'Testing epoch {epoch}...')

        model.eval()

        logits = model(test_samples)

        preds = np.argmax(logits, axis=1)

        # following works for binary labels 0 and 1 specifically
        ssum = preds + test_labels

        true_p = np.argwhere(ssum == 2).flatten()
        true_n = np.argwhere(ssum == 0).flatten()

        tp = len(true_p)
        tn = len(true_n)
        
        false_idxs = np.argwhere(ssum == 1).flatten()

        false = preds[false_idxs]
        fp = np.sum(false)
        fn = len(false) - fp


        f1_score = (2*tp)/(2*tp + fp + fn)

        record_test_metric.append(f1_score)

        print(f'f1 score: {f1_score}')

        if f1_score > max_f1:
            print(f'Got better f1 score at epoch {epoch}, recording model state...')
            max_f1 = f1_score
            optimal_model = deepcopy(nnx.state(model))


    model_name = f"optimal_model_{save_suffix}.p"
    print(f'Saving model {model_name} with f1score {max_f1}...')
    
    dbfile = open(model_name, 'wb')
    pickle.dump(optimal_model, dbfile)
    dbfile.close()

    return record_test_metric, model_layers, model_name
    

