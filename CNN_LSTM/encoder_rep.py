import pandas as pd
import pickle
from EncoderCNN import ResNetEncoder
import torch

combined_RRMs = pd.read_csv('combined_processed_RRM.csv', index_col=0)
# if using CPU
encoder = torch.load('encoder-anneal-True-22-1000.pkl', map_location=lambda storage, loc: storage)
# use line below if running on GPU
# encoder = torch.load('encoder-anneal-True-22-1000.pkl)


vocab = pickle.load(open('vocab2.pkl', 'rb'))
# replace with corresponding vocab pickle

combined_RRMs_input = pd.DataFrame(list(map(lambda word: [vocab(x) for x in word], combined_RRMs.values)))
combined_RRMs_input.index = combined_RRMs.index

new_encoder = ResNetEncoder(84,26, 64)
new_encoder.load_state_dict(encoder)
print('we crunching...')
combined_RRMs_output = new_encoder(torch.autograd.Variable(torch.Tensor(combined_RRMs_input.values.astype(float)).long()))

# line below is not tested (may need tweaking)
pd.DataFrame(combined_RRMs_output.data.numpy()).to_csv('encoder_rep.csv')