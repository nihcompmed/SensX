
# From https://docs.jaxstack.ai/en/latest/JAX_Vision_transformer.html
import jax
import jax.numpy as jnp
from flax import nnx
from copy import deepcopy
import pickle
import vision_transformer as ViT

import numpy as np

import matplotlib.pyplot as plt
from transformers import ViTImageProcessor
from PIL import Image

from torchvision.transforms import v2 as T

img_size = 224

def to_np_array(pil_image):
  return np.asarray(pil_image.convert("RGB"))


def normalize(image):
    # Image preprocessing matches the one of pretrained ViT
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image = image.astype(np.float32) / 255.0
    return (image - mean) / std

tv_test_transforms = T.Compose([
    T.Resize((img_size, img_size)),
    T.Lambda(to_np_array),
    T.Lambda(normalize),
])

###########################################################

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

train_true_imgs = info_dict['train_true']
train_false_imgs = info_dict['train_false']


#train_true_imgs = train_true_imgs[:5000]
#train_false_imgs = train_false_imgs[:5000]


###########################################################

# GET HUPERCUBE BOUNDING THE TRAINING IMAGES

test_batch_size = 5000
collate_images = np.zeros((test_batch_size, img_size, img_size, 3))

train_hypercube_low = np.zeros((img_size, img_size, 3))
train_hypercube_high = np.zeros((img_size, img_size, 3))

total_nn = len(train_true_imgs)

start_batch = 0
end_batch = min(test_batch_size, total_nn)

while start_batch < total_nn:

    print(f'true: {start_batch} to {end_batch}')

    ii = 0

    for pp in range(start_batch, end_batch):

        img = train_true_imgs[pp]
        img_path = f'CelebA/img_align_celeba/{img}'
        image = Image.open(img_path)
        image = tv_test_transforms(image)
        
        collate_images[ii] = image

        ii += 1

    batch_low = np.min(collate_images[:ii], axis=0)
    batch_high = np.max(collate_images[:ii], axis=0)

    train_hypercube_low = np.minimum(batch_low, train_hypercube_low)
    train_hypercube_high = np.maximum(batch_high, train_hypercube_high)

    start_batch += test_batch_size
    end_batch += test_batch_size

    end_batch = min(end_batch, total_nn)


total_nn = len(train_false_imgs)

start_batch = 0
end_batch = min(test_batch_size, total_nn)

while start_batch < total_nn:

    print(f'false: {start_batch} to {end_batch}')

    ii = 0

    for pp in range(start_batch, end_batch):

        img = train_false_imgs[pp]
        img_path = f'CelebA/img_align_celeba/{img}'
        image = Image.open(img_path)
        image = tv_test_transforms(image)
        
        collate_images[ii] = image

        ii += 1

    batch_low = np.min(collate_images[:ii], axis=0)
    batch_high = np.max(collate_images[:ii], axis=0)

    train_hypercube_low = np.minimum(batch_low, train_hypercube_low)
    train_hypercube_high = np.maximum(batch_high, train_hypercube_high)

    start_batch += test_batch_size
    end_batch += test_batch_size

    end_batch = min(end_batch, total_nn)


###########################################################

save_fname = f'category{do_cat}_timestamp{ts}_hypercube.npz'
print(f'Saving {save_fname}...')
np.savez(save_fname, low=train_hypercube_low, high=train_hypercube_high)







