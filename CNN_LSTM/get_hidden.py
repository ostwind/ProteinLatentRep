from __future__ import print_function
import pandas as pd
import torch
import pickle
from EncoderCNN import ResNetEncoder
from decoder import DecoderRNN
from train_decoder import to_var
from torch.utils.data import DataLoader
from preprocessing import preprocess
from RRM_Sequence import RRM_Sequence, collate_fn
from torch.nn.utils.rnn import pack_padded_sequence
# TODO add argparse
"""outputs learned representation of RRM from CNN_LSTM"""
def forward(decoder, features, captions, lengths):
        """Auto-encode RRM sequence vectors."""
        embeddings = decoder.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, (h_n, c_n) = decoder.lstm(packed)
        return h_n[0]

aligned_RRM_path='../data/combined_processed_RRM.csv'
with open('./TrainedModels/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
df_aligned = preprocess(preprocessed=True, RRM_path=aligned_RRM_path,
                               output_path='../data/combined_processed_RRM.csv', vocab=vocab)
loader = RRM_Sequence(df_aligned, vocab)
loader = DataLoader(loader, 16, shuffle=True, collate_fn=collate_fn)

encoderCNN = ResNetEncoder(26, 128).cuda()
decoderRNN = DecoderRNN(128, 256, 26, 1).cuda()
with open('./TrainedModels/encoder-9-551.pkl', 'rb') as encoder:
    encoderCNN.load_state_dict(torch.load(encoder))
with open('./TrainedModels/decoder-9-551.pkl', 'rb') as decoder:
    decoderRNN.load_state_dict(torch.load(decoder))

for batch_idx, (names, rrms_aligned, rrms_unaligned, lengths) in enumerate(loader):
    rrms_aligned = to_var(rrms_aligned) 
    rrms_unaligned = to_var(rrms_unaligned)
    features = encoderCNN(rrms_aligned) 
    hiddens = forward(decoderRNN, features, rrms_unaligned, lengths)
    hiddens = hiddens.data.cpu().numpy()
    if batch_idx == 0:
    	df = pd.DataFrame(hiddens)
    	df['name'] = names
    else:
    	df1 = pd.DataFrame(hiddens)
    	df1['name'] = names
    	df = pd.concat([df, df1])

df.to_csv('hiddens.csv')
