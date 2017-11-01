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
from torch.nn.utils.rnn import pack_padded_sequence  # (?)
from torchvision import transforms  # (?)
from build_vocab import build_vocab
from preprocessing import txt_to_csv, informative_positions
from RRM_Sequence import RRM_Sequence
from decoder import DecoderRNN

# def to_var(x, volatile=False):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x, volatile=volatile)

def main(args):        
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Preprocess the RRM sequence data
    if not os.path.exists(args.info_path):
        # Check that raw data exists and convert to csv
        assert os.path.isfile(args.raw_txt_path), '%s not found!' %(args.raw_txt_path)
        df = txt_to_csv(args.raw_txt_path, args.csv_path)

        # Filter positions that are not in top_n most populated
        df = informative_positions(df, top_n=args.top_n)

        # Save filtered csv for future use
        df.to_csv(args.info_path)
    else:
        print('Using existing csv of informative positions...')
        df = pd.read_csv(args.info_path, index_col=0)

    # Build vocabulary of amino acids
    vocab = build_vocab(df)
    
    BOOM # Stop here, work in progress
    
    # Build data loaders
    train_index = pd.read_csv('../data/train_index.csv',header=None).iloc[:,0]
    train_loader = RRM_Sequence(indices=train_index, info_path=args.info_path)
    train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)
        
    val_index = pd.read_csv('../data/val_index.csv',header=None).iloc[:,0]
    val_loader = RRM_Sequence(indices=val_index, info_path=args.info_path)
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
            
            print('batch_idx % ' % batch_idx)
            print(dic)
            BOOM
            
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)    
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward, and optimize
            decoder.zero_grad()
            # encoder.zero_grad()
            # features = encoder(images)
            features = torch.FloatTensor(embed_size)
            features.fill_(0)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
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
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')    
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--raw_txt_path', type=str, default='../data/PF00076_rp55.txt', 
        help='path for aligned RRM input txt file')
    parser.add_argument('--csv_path', type=str, default='../data/rrm_rp55.csv', 
        help='path for RRM sequence output csv file')
    parser.add_argument('--info_path', type=str, default='../data/rrm_rp55_info.csv', 
        help='path for filtered RRM sequence csv file')    
    parser.add_argument('--top_n', type=int, default=82, 
        help='include top n most populated positions')
#     parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
#                         help='path for vocabulary wrapper')
#     parser.add_argument('--image_dir', type=str, default='./data/resized2014',
#                         help='directory for resized images')
#     parser.add_argument('--caption_path', type=str,
#                         default='./data/annotations/captions_train2014.json',
#                         help='path for train annotation json file')

    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
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

    args = parser.parse_args()    
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    print(args)
    main(args)
    