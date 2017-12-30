# CNN + LSTM autoencoder architecture
### Inspired by the image captioning model found here:
### https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

The scripts in this folder are currently set up to use the combined (labeled + unlabeled) data.

The current default path for this data file is `"../data/combined_data.txt"`.

Each RRM sequence should be formatted as follows:

\>T080824||RNCMPT00434_RRM__0

------YSCKVFLGG--VP--------W--DI-T--------------------------------E----AG---------L-V---N--------------------------T------FR-V---F-------------G----------------S--L-SV------------------------------E----W----------------------------------------------------------------------------------------------------------------P----------------------------------------------------------------------GKDGK-HP------------RC----PPKGYV---Y-----------L-----------V----------F--------E------L--------E-----------K------S--------V--R-------S-------L-------L-QA----------C------------------------------------------------------

If the scripts are run with data in any other format, e.g. with 5 spaces between the RRM name and the sequence instead of a line break, the preprocessing.py script in this folder may need to be changed.

In addition, after preprocessing the data so that there is a processed output file (default path `"../data/aligned_processed_RRM.csv"`), the `train_test_split.py` script should be run with this csv to generate appropriate train/validation/test indices before training. After both these steps, the model can be trained on the already preprocessed data (see `train_decoder.py` for details).

To train the model, run this command locally:

`python train_decoder.py`

Run this command on Prince:

`sbatch train_decoder.s`

The command line argument options are described at the bottom of the `train_decoder.py` script.
