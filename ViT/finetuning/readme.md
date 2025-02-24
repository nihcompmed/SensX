1. Create train/test data sets for a facial attribute: Run `create_celebA_train_test_batches.py`.
   Will save file1=`f'category{do_cat}_traintest{test_split}_timestamp{ts}.p'`.
2. Create training mini-batches: Run `create_training_batches.py`. Takes input file1 and will save file2=`f'category{do_cat}_timestamp{ts}_epochs{n_epochs}_bs{train_batch_size}_bpe{n_batches_per_epoch}.p'`.
3. Fine-tune (balanced): Run `fine_tune.py`. Takes inputs file1 and file2. Will save `finetune_model_state_epoch{epoch}_f1{f1}.p` with the optimal model state.
