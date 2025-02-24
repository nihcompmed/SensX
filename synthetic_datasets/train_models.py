import pickle
import model_library as ml


dbfile = open('synthetic_data.p', 'rb')
data = pickle.load(dbfile)
dbfile.close()

all_log_testf1 = dict()

for dataset in data:

    print(f'Doing dataset {dataset}...')

    load_data = data[dataset]
    
    # Assumes same training hyperparameters
    test_metric, model_layers, model_name = ml.train_save(load_data['X']\
                            , load_data['labels']\
                            , load_data['train_idxs']\
                            , load_data['test_idxs']\
                            , save_suffix=dataset)



    all_log_testf1[dataset] = dict()
    all_log_testf1[dataset]['test_metric'] = test_metric
    all_log_testf1[dataset]['model_layers'] = model_layers
    all_log_testf1[dataset]['model_name'] = model_name


dbfile = open('training_log_testf1.p', 'wb')
pickle.dump(all_log_testf1, dbfile)
dbfile.close()
