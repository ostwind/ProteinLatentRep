from __future__ import print_function

# Build vocabulary list of amino acids for CNN+LSTM autoencoder architecture
# Adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py




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
    """Builds vocabulary of amino acids that appear in dataframe"""
    
    print('Building vocabulary of amino acids...')
    
    # Create vocab wrapper
    vocab = Vocabulary()
    
    # Add words (amino acids) from RRM sequences
    for word in df.values.flat:
        if word:  # Ignore any that are None
            vocab.add_word(word.upper())  # Convert to upper case
    
    # Add special tokens
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    
    return vocab
