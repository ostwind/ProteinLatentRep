import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def validate(val_loader, encoder, decoder, criterion):
    """Returns validation loss for DataLoader val_loader, trained encoder, trained decoder,
    and criterion loss function."""
    for batch_idx, (names, rrms_aligned, rrms_unaligned, lengths) in enumerate(val_loader):
        rrms_aligned = to_var(rrms_aligned) 
        rrms_unaligned = to_var(rrms_unaligned)
        targets = pack_padded_sequence(rrms_unaligned, lengths, batch_first=True)[0]

        # Get outputs for validation sequences
        features = encoder(rrms_aligned)
        outputs = decoder(features, rrms_unaligned, lengths)

        # Return loss according to given criterion
        loss = criterion(outputs, targets)
        return loss.data[0]
    
def early_stop(prev_valid_loss, val_loader, encoder, decoder, criterion):
    """Returns Boolean indicating whether we should stop early based on validation loss (also returned)"""
    valid_loss = validate(val_loader, encoder, decoder, criterion)
    
    if valid_loss > prev_valid_loss:
        return (True, valid_loss)
    return (False, valid_loss)
