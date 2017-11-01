# Train the LSTM decoder for the CNN+LSTM autoencoder architecture

# Adapted from:
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/train.py

# TODO:
# Add preprocessing (IN PROGRESS)
# Replace vocab
# Replace data_loader and for loop that iterates over training data
# Integrate with CNN encoder (Currently using dummy features to feed into LSTM)
# Use a validation set to do any hyperparameter tuning (See below)
# Play around with model params (Also review other hyperparams we could tune)

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from preprocessing import informative_positions
from decoder import DecoderRNN
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# def to_var(x, volatile=False):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x, volatile=volatile)

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Preprocess the RRM sequence data
    raw_txt_path = '../data/PF00076_rp55.txt'  # Path for aligned RRM input txt file
    top_n = 82  # Include top n most populated positions
    
    # Check that raw data exists and convert to csv
    assert os.path.isfile(raw_txt_path), '%s not found!' %(raw_txt_path)
    df = txt_to_csv(raw_txt_path, csv_path)
    
    # Filter positions that are not in top_n most populated
    df = informative_positions(df, top_n=N)
    
    from IPython.display import display  # DEBUGGING
    display(df)  # DEBUGGING
    BOOM  # DEBUGGING
    
#     # Image preprocessing
#     # For normalization, see https://github.com/pytorch/vision#models
#     transform = transforms.Compose([ 
#         transforms.RandomCrop(args.crop_size),
#         transforms.RandomHorizontalFlip(), 
#         transforms.ToTensor(), 
#         transforms.Normalize((0.485, 0.456, 0.406), 
#                              (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    
    # Build the models
    # encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if args.cuda:
#     if torch.cuda.is_available():
#         encoder.cuda()
        decoder.cuda()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) #+ list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
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
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
#     parser.add_argument('--crop_size', type=int, default=224 ,
#                         help='size for randomly cropping images')
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
    
    # Model parameters
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
    print(args)
    main(args)