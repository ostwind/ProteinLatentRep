# Build vocabulary list of amino acids for LSTM decoder
# Adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py

import argparse
import pickle
import numpy as np
import pandas as pd
from collections import Counter

class Vocabulary(object):

    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(df):

    """Builds vocabulary of amino acids that appear 
    in csv at csv_path"""
    
    print('Building vocabulary of amino acids...')
    
    # Create vocab wrapper
    vocab = Vocabulary()
    
    # Add words from RRM sequences
    for word in df.values.flat: 
    # Each "word" is a single amino acid
        vocab.add_word(word)
    
    # Add special tokens
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    return vocab
