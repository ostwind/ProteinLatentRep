# Train the CNN+LSTM autoencoder architecture
# Adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/train.py

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from preprocessing import preprocess
from Decoder import DecoderRNN
from EncoderCNN import ResNetEncoder
from RRM_Sequence import RRM_Sequence, collate_fn

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):   
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Preprocess the RRM data
    vocab, df_aligned = preprocess(preprocessed=args.preprocessed, 
        RRM_path=args.aligned_RRM_path, output_path=args.processed_RRM_path)

    # Prepare the training and validation sets
    train_index = pd.read_csv('../data/train_index.csv',header=None).iloc[:,0]
    train_loader = RRM_Sequence(df_aligned.loc[train_index, :], vocab)
    train_loader = DataLoader(train_loader, batch_size=args.batch_size, 
        shuffle=True, collate_fn=collate_fn)
    
    val_index = pd.read_csv('../data/val_index.csv',header=None).iloc[:,0]
    val_loader = RRM_Sequence(df_aligned.loc[val_index, :], vocab)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, 
        shuffle=True, collate_fn=collate_fn)

    # Define the models
    encoder = ResNetEncoder(len(vocab), args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        for batch_idx, (names, rrms_aligned, rrms_unaligned, lengths) in enumerate(train_loader):
            # Set mini-batch dataset
            rrms_aligned = to_var(rrms_aligned) 
            rrms_unaligned = to_var(rrms_unaligned)
            targets = pack_padded_sequence(rrms_unaligned, lengths, batch_first=True)[0]
            
            
            # Forward, backward, and optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(rrms_aligned) 
            outputs = decoder(features, rrms_unaligned, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print log info
            if batch_idx % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, batch_idx, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            if (batch_idx+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, batch_idx+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, batch_idx+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')    

    # File paths
    parser.add_argument('--model_path', type=str, default='./',
                        help='path for saving trained models')
    parser.add_argument('--aligned_RRM_path', type=str, default='../data/PF00076_rp55.txt', 
                        help='path for aligned RRM input file')
    parser.add_argument('--processed_RRM_path', type=str, default='../data/aligned_processed_RRM.csv', 
                        help='path for outputting processed aligned_RRM data')

    # Preprocessing settings
    parser.add_argument('--preprocessed', action='store_true', default=False,
                        help='if RRM file is preprocessed')
    # TODO: double check the store_true action
    parser.add_argument('--top_n', type=int, default=82, 
                        help='include top n most populated positions')

    # Control printing/saving
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    
    # Model hyperparameters
    parser.add_argument('--embed_size', type=int, default=128, #256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=256, #512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()    
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)

    