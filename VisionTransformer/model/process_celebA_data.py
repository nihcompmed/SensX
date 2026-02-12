from PIL import Image
import pickle

def parser_labels(line):
    line = line.strip(' \n')
    line = line.split(' ')
    return [x for x in line if x != '']

labels_file = 'CelebA/list_attr_celeba.txt'
ll = open(labels_file, 'r')
header = parser_labels(ll.readline())

labels_dict = dict()

for line in ll:

    info = parser_labels(line)
    img_name = info[0]
    labels_dict[img_name] = dict()
    for tt, val in zip(header[1:], info[1:]):
        val = int(val)
        if val == -1:
            val = 0
        labels_dict[img_name][tt] = int(val)

ll.close()



dbfile = open('CelebA_img_labels.p', 'wb')
pickle.dump(labels_dict, dbfile)
dbfile.close()


