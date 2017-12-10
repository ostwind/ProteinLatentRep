import pandas as pd
import pickle
from EncoderCNN import ResNetEncoder
import torch

combined_RRMs = pd.read_csv('combined_processed_RRM.csv', index_col=0)

if not torch.cuda.is_available():
	encoder = torch.load('encoder-anneal-True-22-1000.pkl', map_location=lambda storage, loc: storage)
else:
	encoder = torch.load('encoder-anneal-True-22-1000.pkl')


vocab = pickle.load(open('vocab2.pkl', 'rb'))
# replace with corresponding vocab pickle

combined_RRMs_input = pd.DataFrame(list(map(lambda word: [vocab(x) for x in word], combined_RRMs.values)))
combined_RRMs_input.index = combined_RRMs.index

new_encoder = ResNetEncoder(84,26, 64)
if torch.cuda.is_available():
	new_encoder = new_encoder.cuda()

new_encoder.load_state_dict(encoder)

print('we crunching...')

if not torch.cuda.is_available():
	combined_RRMs_output = new_encoder(torch.autograd.Variable(torch.Tensor(combined_RRMs_input.values.astype(float)).long()))
else:
	combined_RRMs_output = new_encoder(torch.autograd.Variable(torch.Tensor(combined_RRMs_input.values.astype(float)).long()).cuda())

# code below is not tested (may need tweaking)
if not torch.cuda.is_available():
	combined_RRMs_output = pd.DataFrame(combined_RRMs_output.data.numpy())
else:
	combined_RRMs_output = pd.DataFrame(combined_RRMs_output.data.cpu().numpy())
combined_RRMs_output.index = combined_RRMs.index
combined_RRMs_output.to_csv('encoder_rep.csv')