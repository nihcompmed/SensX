import numpy as np
import pickle
import math
import time

dbfile = open('CelebA_img_labels.p', 'rb')
labels_dict = pickle.load(dbfile)
dbfile.close()

#for kk in labels_dict:
#    print(labels_dict[kk].keys())
#    break
#'5_o_Clock_Shadow',\ 'Arched_Eyebrows',\ 'Attractive',\ 'Bags_Under_Eyes',\ 'Bald', \
#'Bangs',\ 'Big_Lips',\ 'Big_Nose',\ 'Black_Hair',\ 'Blond_Hair',\
#'Blurry',\ 'Brown_Hair',\ 'Bushy_Eyebrows',\ 'Chubby',\ 'Double_Chin',\
#'Eyeglasses',\ 'Goatee',\ 'Gray_Hair',\ 'Heavy_Makeup',\ 'High_Cheekbones',\
#'Male',\ 'Mouth_Slightly_Open',\ 'Mustache',\ 'Narrow_Eyes',\ 'No_Beard',\
#'Oval_Face',\ 'Pale_Skin',\ 'Pointy_Nose',\ 'Receding_Hairline',\ 'Rosy_Cheeks',\
#'Sideburns',\ 'Smiling',\ 'Straight_Hair',\ 'Wavy_Hair',\ 'Wearing_Earrings',\
#'Wearing_Hat',\ 'Wearing_Lipstick',\ 'Wearing_Necklace',\ 'Wearing_Necktie',\ 'Young'

# do_cat = 'Smiling'
#do_cat = 'Eyeglasses'
#do_cat = 'Wearing_Hat'
do_cat = 'Eyeglasses'

# Binary classes
truecat = []
falsecat = []

for img in labels_dict:

    if labels_dict[img][do_cat]:
        truecat.append(img)
    else:
        falsecat.append(img)

truecat = np.array(truecat)
falsecat = np.array(falsecat)

test_split = 0.2

n_true = len(truecat)
n_false = len(falsecat)

n_true_test = math.floor(test_split * n_true)
n_false_test = math.floor(test_split * n_false)


true_mask = np.ones(n_true, dtype=bool)
false_mask = np.ones(n_false, dtype=bool)


# randomly pick idxs for training/test
true_test_idxs = np.random.choice(n_true, size=n_true_test, replace=False)
true_mask[true_test_idxs] = 0

false_test_idxs = np.random.choice(n_false, size=n_false_test, replace=False)
false_mask[false_test_idxs] = 0

train_true_imgs = truecat[true_mask]
test_true_imgs = truecat[~true_mask]

train_false_imgs = falsecat[false_mask]
test_false_imgs = falsecat[~false_mask]

info_dict = dict()

info_dict['train_true'] = train_true_imgs
info_dict['test_true'] = test_true_imgs

info_dict['train_false'] = train_false_imgs
info_dict['test_false'] = test_false_imgs


ts = int(time.time())

save_fname = f'category{do_cat}_traintest{test_split}_timestamp{ts}.p'
dbfile = open(save_fname, 'wb')
pickle.dump(info_dict, dbfile)
dbfile.close()


