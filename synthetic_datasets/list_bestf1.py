import pickle
import numpy as np


dbfile = open('training_log_testf1.p', 'rb')
data = pickle.load(dbfile)
dbfile.close()

for dataset in data:

    test_metrics = np.array(data[dataset]['test_metric'])

    print(dataset, np.round(np.amax(test_metrics)*100,2))

