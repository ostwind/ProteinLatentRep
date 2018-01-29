# Adapted from Sean Robertson's "Translation with a Sequence to Sequence Network and Attention" tutorial:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

from __future__ import print_function, division
import os
import argparse
import pickle
import random
import time
import math
#import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from preprocessing import preprocess
#from early_stopping import validate, early_stop
from train_test_split import train_test_split
from model import EncoderRNN, DecoderRNN

##########
use_cuda = torch.cuda.is_available()
SOS_token = '<start>'
EOS_token = '<end>'
teacher_forcing_ratio = 0.5
##########

def asMinutes(s):
    '''Helper function to convert seconds to minutes'''
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    '''Helper function to print time elapsed/remaining in minutes'''
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def to_var(x, volatile=False):
    '''Simple wrapper for PyTorch's Variable'''
    if use_cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    ##########
    # Make a directory to save models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Preprocess the RRM data
    vocab, df_aligned = preprocess(preprocessed=args.preprocessed, RRM_path=args.aligned_RRM_path,
                                   output_path=args.processed_RRM_path, sep=args.sep)
    df_aligned = train_test_split(df_aligned)
    with open(os.path.join(args.model_path, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
        
    # Prepare the training and validation sets
    train_index = pd.read_csv('../data/train_index.csv',header=None).iloc[:,0]
    train_loader = RRM_Sequence(df_aligned.loc[train_index, :], vocab)
    train_loader = DataLoader(train_loader, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    
    val_index = pd.read_csv('../data/val_index.csv',header=None).iloc[:,0]
    val_loader = RRM_Sequence(df_aligned.loc[val_index, :], vocab)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    ##########
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: Use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length
