# Affinity Regression


To run affinity regression:
```
python run_training.py --emb_file=10ep_12noise_15conv_65k.csv --lr 0.000001 --optim=adam --print_every 50 --eval_every 100 --batch_size 1
```

Make sure that these arguments are present and agree. The values below for proteinfeatures and file_2 are the default parameters. This last argument should direct to the file from the original dropbox_files.

- emb_file: path to embedding file
- proteinfeatures='Proteinsequence_latentspacevectors.txt'
- file_2='../dropbox_files/Full_predictableRRM_pearson_gt3_trainset216_test24.txt'

```
Namespace(batch_size=1, dtype='--z', emb_file='10ep_12noise_15conv_65k.csv', eval_every=100, file_2='../dropbox_files/Full_predictableRRM_pearson_gt3_trainset216_test24.txt', loss='SGD', lr=1e-06, num_epochs=2000, optim='adam', print_every=50, probefeatures='ones', profiles='../dropbox_files/Fullset_z_scores_setAB.txt', proteinfeatures='Proteinsequence_latentspacevectors.txt')
```

Code to evaluate predictive results of model, updated to use pytorch


