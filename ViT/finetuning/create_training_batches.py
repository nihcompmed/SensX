import numpy as np
import pickle
import math
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# do_cat = 'Smiling'
# test_split = 0.2
# ts = 1737675700

do_cat = 'Eyeglasses'
test_split = 0.2
ts = 1737861746

fname = f'category{do_cat}_traintest{test_split}_timestamp{ts}.p'

dbfile = open(fname, 'rb')
info_dict = pickle.load(dbfile)
dbfile.close()

train_true_imgs = info_dict['train_true']
test_true_imgs=info_dict['test_true']

train_false_imgs=info_dict['train_false']
test_false_imgs=info_dict['test_false']



# Create balanced batch
def create_balanced_batch(seta, labela, setb, labelb, batch_size):

    # na = len(seta)
    # nb = len(setb)

    # bala = na/(na+nb)

    # n_batch_a = math.floor(bala * batch_size)
    # n_batch_b = batch_size - n_batch_a

    # assert min(n_batch_a, n_batch_b) > 0

    n_batch_a = int(batch_size/2)
    n_batch_b = batch_size - n_batch_a
    

    batcha = np.random.choice(seta, size=n_batch_a, replace=False)
    labelsa = np.ones(n_batch_a, dtype=int)*labela

    batchb = np.random.choice(setb, size=n_batch_b, replace=False)
    labelsb = np.ones(n_batch_b, dtype=int)*labelb


    this_batch = np.hstack([batcha, batchb])
    this_labels = np.hstack([labelsa, labelsb])

    # Shuffle
    idxs = np.arange(batch_size)
    np.random.shuffle(idxs)

    this_batch = this_batch[idxs]
    this_labels = this_labels[idxs]

    return this_batch, this_labels


train_batch_size = 25
n_batches_per_epoch = 1000
n_epochs = 100

all_batches = []

for epoch in tqdm(range(n_epochs)):

    all_batches.append([])

    for bb in range(n_batches_per_epoch):

        batch_imgs, batch_labels =\
                create_balanced_batch(train_true_imgs, 1, train_false_imgs, 0, train_batch_size)

        # uu, vv = np.unique(batch_labels, return_counts=True)
        # print(uu, vv)
        # exit()

        all_batches[-1].append((batch_imgs, batch_labels))

save_fname = f'category{do_cat}_timestamp{ts}_epochs{n_epochs}_bs{train_batch_size}_bpe{n_batches_per_epoch}.p'
dbfile = open(save_fname, 'wb')
pickle.dump(all_batches, dbfile)
dbfile.close()


#uu, ll = np.unique(batch_labels, return_counts=True)
#print(uu, ll)
#
#for img, img_label in zip(batch_imgs, batch_labels):
#
#    img_path = f'CelebA/Img/img_align_celeba/{img}'
#
#    image = Image.open(img_path)
#
#    plt.imshow(image)
#
#    plt.title(img_label)
#
#    plt.show()






