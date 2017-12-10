from __future__ import print_function
import pandas as pd
import torch
import pickle
import argparse
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

def main(args):
    """Output learned representation of RRMs from CNN_LSTM autoencoder."""
    
    # Load pickled vocab
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Load the already preprocessed data
    df_aligned = preprocess(preprocessed=True, RRM_path=args.processed_RRM_path,
                                   output_path=args.processed_RRM_path, vocab=vocab)
    
    # Data loader
    loader = RRM_Sequence(df_aligned, vocab)
    loader = DataLoader(loader, 16, shuffle=True, collate_fn=collate_fn)

    encoderCNN = ResNetEncoder(84, 26, 64)  # TODO don't hardcode?
    decoderRNN = DecoderRNN(64, 128, 26, 1)  # TODO don't hardcode?
    
    # Use CUDA if available
    if torch.cuda.is_available():
        encoderCNN.cuda()
        decoderRNN.cuda()
    
    # Load pickled models
    with open(args.encoder_path, 'rb') as encoder:
        encoderCNN.load_state_dict(torch.load(encoder))
    with open(args.decoder_path, 'rb') as decoder:
        decoderRNN.load_state_dict(torch.load(decoder))

    # Loop over data
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

    # Write to file
    df.to_csv(args.hidden_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # File paths
    parser.add_argument('--vocab_path', type=str, default='./TrainedModels/vocab.pkl',
                        help='path for pickled vocab')
    parser.add_argument('--processed_RRM_path', type=str, default='../data/combined_processed_RRM.csv',
                        help='path for preprocessed aligned_RRM data')
    parser.add_argument('--encoder_path', type=str, default='./TrainedModels/encoder-anneal-True-22-1000.pkl',
                        help='path for saved trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./TrainedModels/decoder-anneal-True-22-1000.pkl',
                        help='path for saved trained decoder')
    parser.add_argument('--hidden_path', type=str, default='./TrainedModels/hiddens.csv',
                       help='path to save hidden representations')

    args = parser.parse_args()
    
    # CUDA settings
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
    