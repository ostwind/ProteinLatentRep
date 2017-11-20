"""outputs learned representation of RRM from CNN_LSTM"""

import pandas as pd
import pickle
import torch
from EncoderCNN import ResNetEncoder
from decoder import DecoderRNN
from train_decoder import to_var
from torch.utils.data import DataLoader
from preprocessing import preprocess
from RRM_Sequence import RRM_Sequence, collate_fn
from torch.nn.utils.rnn import pack_padded_sequence

def forward(decoder, features, captions, lengths):
        """Auto-encode RRM sequence vectors."""
        embeddings = decoder.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, (h_n, c_n) = decoder.lstm(packed)
        return h_n[0]

aligned_RRM_path="../data/aligned_processed_RRM.csv"
vocab, df_aligned = preprocess(preprocessed=True, RRM_path=aligned_RRM_path,  
                               output_path='../data/aligned_processed_RRM.csv')
val_index = pd.read_csv('../data/val_index.csv',header=None).iloc[:,0]
val_loader = RRM_Sequence(df_aligned.loc[val_index, :], vocab)
val_loader = DataLoader(val_loader, 16, shuffle=True, collate_fn=collate_fn)

encoderCNN = ResNetEncoder(26, 128).cuda()
decoderRNN = DecoderRNN(128, 256, 26, 1).cuda()
with open('./TrainedModels/encoder-9-551.pkl', 'rb') as encoder:
    encoderCNN.load_state_dict(torch.load(encoder))
with open('./TrainedModels/decoder-9-551.pkl', 'rb') as decoder:
    decoderRNN.load_state_dict(torch.load(decoder))

for batch_idx, (names, rrms_aligned, rrms_unaligned, lengths) in enumerate(val_loader):
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
