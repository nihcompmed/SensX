# https://docs.jaxstack.ai/en/latest/JAX_Vision_transformer.html

import numpy as np
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
import pickle

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import nnx
import flax

import vision_transformer as ViT

from PIL import Image
import time

import math

img_size = 224

#############################################################
# training and testing image preprocessing methods
#############################################################

def to_np_array(pil_image):
  return np.asarray(pil_image.convert("RGB"))


def normalize(image):
    # Image preprocessing matches the one of pretrained ViT
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image.astype(np.float64) / 255.0
    return (image - mean) / std


tv_train_transforms = T.Compose([
    T.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.Lambda(to_np_array),
    T.Lambda(normalize),
])


tv_test_transforms = T.Compose([
    T.Resize((img_size, img_size)),
    T.Lambda(to_np_array),
    T.Lambda(normalize),
])


###########################################################
# Initialize model
###########################################################

model = ViT.VisionTransformer(num_classes=1000)

###########################################################
# Load model state
###########################################################

load_model_name = f"vit-base-patch16-224_model_state.p"
dbfile = open(load_model_name, 'rb')
model_state = pickle.load(dbfile)
dbfile.close()

nnx.update(model, model_state)

###########################################################
# Define custom classifier to finetune with new image labeled data
###########################################################

n_classes = 2

model.classifier = nnx.Linear(model.classifier.in_features, n_classes, rngs=nnx.Rngs(0))

#################################################
# Model training
#################################################

# Setting up optimizer

import optax

learning_rate = 0.00001
momentum = 0.8

############

# MANU : Changing to constant learning rate for now
#total_steps = len(train_dataset) // train_batch_size
#lr_schedule = optax.linear_schedule(learning_rate, 0.0, num_epochs * total_steps)
lr_schedule = optax.constant_schedule(learning_rate)

optimizer = nnx.Optimizer(model, optax.sgd(lr_schedule, momentum, nesterov=True))

#############################################


# Loss function

def compute_losses_and_logits(model: nnx.Module, images: jax.Array, labels: jax.Array):
    logits = model(images)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits

##############################################

# Setting up training and evaluation functions

@nnx.jit
def train_step(model, optimizer, images, labels):

    # MANU
        #model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, np.ndarray]

    ## Convert np.ndarray to jax.Array on GPU
    #images = jnp.array(batch["image"])
    #labels = jnp.array(batch["label"], dtype=jnp.int32)

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)

    optimizer.update(grads)  # In-place updates.

    return loss


@nnx.jit
def eval_step(
    model: nnx.Module, batch: dict[str, np.ndarray], eval_metrics: nnx.MultiMetric
):
    # Convert np.ndarray to jax.Array on GPU
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"], dtype=int)
    loss, logits = compute_losses_and_logits(model, images, labels)

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )


eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
)


train_metrics_history = {
    "train_loss": [],
}

eval_metrics_history = {
    "val_loss": [],
    "val_accuracy": [],
}



from tqdm import tqdm


bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"


def train_one_epoch(epoch, batches, train_batch_size, img_size):
    model.train()  # Set model to the training mode: e.g. update batch statistics

    #with tqdm.tqdm(
    #    desc=f"[train] epoch: {epoch}/{n_epochs}, ",
    #    total=len(batches),
    #    bar_format=bar_format,
    #    leave=True,
    #) as pbar:
    for batch in tqdm(batches):

        #tic = time.time()

        batch_images, batch_labels = batch

        collate_images = np.zeros((train_batch_size, img_size, img_size, 3))

        #toc = time.time()
        #print(f'\ntime taken to set up collate {toc-tic}')

        #tic = time.time()

        for ii in range(train_batch_size):
            img_path = f'CelebA/img_align_celeba/{batch_images[ii]}'
            image = Image.open(img_path)
            image = tv_train_transforms(image)
            collate_images[ii] = image


        #toc = time.time()
        #print(f'time taken to collate {toc-tic}')

        #tic = time.time()

        batch_labels = jnp.array(batch_labels, dtype=int)

        collate_images = jnp.array(collate_images)

        #toc = time.time()
        #print(f'time taken to jnp data {toc-tic}')

        #tic = time.time()

        loss = train_step(model, optimizer, collate_images, batch_labels)

        #jax.block_until_ready(0)
        #toc = time.time()
        #print(f'time taken to compute loss {toc-tic}')

        #train_metrics_history["train_loss"].append(loss.item())
        #pbar.set_postfix({"loss": loss.item()})
        #pbar.update(1)

    return loss.item()

def evaluate_model(epoch):
    # Compute the metrics on the train and val sets after each training epoch.
    model.eval()  # Set model to evaluation model: e.g. use stored batch statistics

    eval_metrics.reset()  # Reset the eval metrics
    for val_batch in val_loader:
        eval_step(model, val_batch, eval_metrics)

    for metric, value in eval_metrics.compute().items():
        eval_metrics_history[f'val_{metric}'].append(value)

    print(f"[val] epoch: {epoch + 1}/{num_epochs}")
    print(f"- total loss: {eval_metrics_history['val_loss'][-1]:0.4f}")
    print(f"- Accuracy: {eval_metrics_history['val_accuracy'][-1]:0.4f}")



##############################################


# Load training batches info

# do_cat = 'Smiling'
# ts = 1737675700
# test_split = 0.2

do_cat = 'Eyeglasses'
test_split = 0.2
ts = 1737861746


load_fname = f'category{do_cat}_traintest{test_split}_timestamp{ts}.p'
dbfile = open(load_fname, 'rb')
info_dict = pickle.load(dbfile)
dbfile.close()

test_true_imgs = info_dict['test_true']
test_false_imgs = info_dict['test_false']




train_batch_size = 25
n_batches_per_epoch = 1000
n_epochs = 100

load_fname = f'category{do_cat}_timestamp{ts}_epochs{n_epochs}_bs{train_batch_size}_bpe{n_batches_per_epoch}.p'
dbfile = open(load_fname, 'rb')
all_batches = pickle.load(dbfile)
dbfile.close()

# Model trainining

test_batch_size = 250
collate_images = np.zeros((test_batch_size, img_size, img_size, 3))

max_acc = -math.inf

for epoch in range(n_epochs):

    epoch_loss = train_one_epoch(epoch, all_batches[epoch], train_batch_size, img_size)

    # Test
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    nn = 0

    model.eval()


    total_nn = len(test_true_imgs)

    start_batch = 0
    end_batch = min(test_batch_size, total_nn)

    while start_batch < total_nn:

        print(f'true: {start_batch} to {end_batch}')

        ii = 0

        for pp in range(start_batch, end_batch):

            img = test_true_imgs[pp]
            img_path = f'CelebA/img_align_celeba/{img}'
            image = Image.open(img_path)
            image = tv_test_transforms(image)
            
            collate_images[ii] = image

            ii += 1

        logits = model(jnp.array(collate_images[:ii]))
        softmax = jax.nn.softmax(logits, axis=1)
        preds = jnp.argmax(softmax, axis=1)

        this_tp = jnp.sum(preds)
        this_nn = len(preds)
        this_fn = this_nn - this_tp

        nn += this_nn
        tp += this_tp
        fn += this_fn

        start_batch += test_batch_size
        end_batch += test_batch_size

        end_batch = min(end_batch, total_nn)


    total_nn = len(test_false_imgs)

    start_batch = 0
    end_batch = min(test_batch_size, total_nn)

    while start_batch < total_nn:

        print(f'false: {start_batch} to {end_batch}')

        ii = 0

        for pp in range(start_batch, end_batch):

            img = test_false_imgs[pp]
            img_path = f'CelebA/img_align_celeba/{img}'
            image = Image.open(img_path)
            image = tv_test_transforms(image)
            
            collate_images[ii] = image

            ii += 1

        logits = model(jnp.array(collate_images[:ii]))
        softmax = jax.nn.softmax(logits, axis=1)
        preds = jnp.argmax(softmax, axis=1)

        this_fp = jnp.sum(preds)
        this_nn = len(preds)
        this_tn = this_nn - this_fp

        nn += this_nn
        fp += this_fp
        tn += this_tn

        start_batch += test_batch_size
        end_batch += test_batch_size

        end_batch = min(end_batch, total_nn)



    f1 = 2*tp/(2*tp + fp + fn)

    print(f'epoch {epoch}: loss {epoch_loss} : f1 {f1}')

    if f1 > max_acc:

        save_model_name = f"finetune_model_state_epoch{epoch}_f1{f1}.p"
        
        print(f'Saving model {save_model_name}...')

        dbfile = open(save_model_name, 'wb')
        pickle.dump(nnx.state(model), dbfile)
        dbfile.close()



#    evaluate_model(epoch)
#
#
#
#plt.plot(train_metrics_history["train_loss"], label="Loss value during the training")
#plt.legend()











