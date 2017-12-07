from __future__ import print_function
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def validate(val_loader, encoder, decoder, criterion):
    """Returns validation loss for DataLoader val_loader, trained encoder, trained decoder,
    and criterion loss function."""
    average_loss = 0
    for batch_idx, (names, rrms_aligned, rrms_unaligned, lengths) in enumerate(val_loader):
        rrms_aligned = to_var(rrms_aligned) 
        rrms_unaligned = to_var(rrms_unaligned)
        targets = pack_padded_sequence(rrms_unaligned, lengths, batch_first=True)[0]

        # Get outputs for validation sequences
        features = encoder(rrms_aligned)
        outputs = decoder(features, rrms_unaligned, lengths)

        # Return loss according to given criterion
        loss = criterion(outputs, targets)
        average_loss += loss.data[0]
    average_loss /= len(val_loader)
    return average_loss
    
def early_stop(val_acc_history, k=10, required_progress=1e-4):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by 
        at least required_progress amount to be non-trivial
    @param t: number of training steps 
    @return: a boolean indicates if the model should earily stop
    """
    non_trivial = 1
    if len(val_acc_history)>=k+1:
        non_trivial = 0
        for i, acc in enumerate(val_acc_history):
            if i != len(val_acc_history) - 1:
                if val_acc_history[i+1]-acc < -required_progress:
                    non_trivial = 1
                    break
    return not non_trivial
