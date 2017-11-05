# Train the LSTM decoder for the CNN+LSTM autoencoder architecture
# Adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/train.py

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from torch.nn.utils.rnn import pack_padded_sequence  
from build_vocab import build_vocab
from preprocessing import txt_to_csv, informative_positions, encode_output
from RRM_Sequence import RRM_Dataset #, RRM_OriginalSequence, RRM_AlignedSequence
from decoder import DecoderRNN
# from EncoderCNN import ResNetEncoder

def main(args):        
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Preprocess the RRM sequence data (if necessary)
    if not os.path.exists(args.info_path):
        # # Check that **aligned** data exists and convert to csv
        # # **use either aligned or unaligned**
        # print('Using aligned sequences...')
        # assert os.path.isfile(args.aligned_txt_path), '%s not found!' %(args.aligned_txt_path)
        # df = txt_to_csv(args.aligned_txt_path, args.csv_path)

        # Check that **unaligned** data exists and convert to csv
        # **use either aligned or unaligned**
        print ('Using unaligned sequences...')
        assert os.path.isfile(args.raw_txt_path), '%s not found!' %(args.raw_txt_path)
        df = txt_to_csv(args.raw_txt_path, args.csv_path)

#         # Filter positions that are not in top_n most populated
#         # **don't need to do this if using unaligned**
#         df = informative_positions(df, top_n=args.top_n)

#         # Save filtered csv for future use
#         # **don't need to do this if using unaligned**
#         df.to_csv(args.info_path)

    else:
        # Load saved csv
        print('Using aligned sequences...')        
        print('Using existing csv of informative positions...')
        df = pd.read_csv(args.info_path, index_col=0)

    # One-hot encode filtered RRM sequences (if necessary)
    # **will be either aligned or unaligned depending on above**
    if not os.path.exists(args.encoder_path):        
        encode_output(df, args.encoder_path, args.label_encoder_path, root_dir=args.encoded_output_path)
        
    # Build vocabulary of amino acids
    vocab = build_vocab(df)
        
    # Build data loaders
    train_index = pd.read_csv('../data/train_index.csv',header=None).iloc[:,0]
    train_loader = RRM_Dataset(indices=train_index, root_dir=args.encoded_output_path)
    train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)
        
    val_index = pd.read_csv('../data/val_index.csv',header=None).iloc[:,0]
    val_loader = RRM_Dataset(indices=val_index, root_dir=args.encoded_output_path)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=True)
    
    # Build the models
    # encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if args.cuda:
        # encoder.cuda()
        decoder.cuda()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) #+ list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        for batch_idx, dic in enumerate(train_loader):
        #for i, (images, captions, lengths) in enumerate(train_loader):
            # Set mini-batch dataset
            data = Variable(dic['seq'].view(-1, 3608))
                        
            # Forward, backward, and optimize
            decoder.zero_grad()
            # encoder.zero_grad()
            # features = encoder(data)
            features = torch.FloatTensor(args.embed_size)  # Dummy features for now
            features.fill_(0)  # Dummy features for now
            outputs = decoder(features, data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
#                 torch.save(encoder.state_dict(), 
#                            os.path.join(args.model_path, 
#                                         'encoder-%d-%d.pkl' %(epoch+1, i+1)))

if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # CUDA settings
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')    
    
    # File paths
    parser.add_argument('--model_path', type=str, default='./',
                        help='path for saving trained models')
    parser.add_argument('--aligned_txt_path', type=str, default='../data/PF00076_rp55.txt', 
        help='path for aligned RRM input txt file')
    parser.add_argument('--raw_txt_path', type=str, default='../data/PF00076_rp55-2.txt', 
        help='path for unaligned RRM input txt file')
    parser.add_argument('--csv_path', type=str, default='../data/rrm_rp55.csv', 
        help='path for RRM sequence output csv file')
    parser.add_argument('--info_path', type=str, default='../data/rrm_rp55_info.csv', 
        help='path for filtered RRM sequence csv file')    
    parser.add_argument('--top_n', type=int, default=82, 
        help='include top n most populated positions')
    parser.add_argument('--encoder_path', type=str, default='../data/onehot.p', 
        help='path for one-hot encoder')
    parser.add_argument('--label_encoder_path', type=str, default='../data/label_encoder.p', 
        help='path for one-hot encoder')
    parser.add_argument('--encoded_output_path', type=str, default='../data', 
        help='path for one-hot encoded individual RRM output csv files')
    
    # Intervals for writing/saving progress while training
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    
    # Model hyperparameters
    parser.add_argument('--embed_size', type=int, default=4, #256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=8, #512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # Parse the command line args
    args = parser.parse_args()    
    
    # More CUDA settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    print(args)
    main(args)
    