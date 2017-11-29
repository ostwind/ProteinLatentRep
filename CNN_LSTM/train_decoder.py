from __future__ import print_function
import os
import argparse
import torch
import pickle
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from preprocessing import preprocess
from decoder import DecoderRNN
from EncoderCNN import ResNetEncoder
from RRM_Sequence import RRM_Sequence, collate_fn
from early_stopping import validate, early_stop
from train_test_split import train_test_split
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def adjust_learning_rate(optimizer, epoch):
    """Decays learning rate to a quarter each epoch, after first 5 epochs"""
    if epoch > 5:
        lr = args.learning_rate * (.5 ** ((epoch-5)//5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main(args):   
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

    # Define the models
    encoder = ResNetEncoder(df_aligned.shape[1], len(vocab), args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
            
    # Train the models
    total_step = len(train_loader)
    val_loss_history = []
    for epoch_num, epoch in enumerate(range(args.num_epochs)):
        for batch_idx, (names, rrms_aligned, rrms_unaligned, lengths) in enumerate(train_loader):
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
            if (batch_idx+1) % args.log_step == 0:
                val_loss = validate(val_loader, encoder, decoder, criterion)
                val_loss_history.append(val_loss)
                print('Epoch [%d/%d], Step [%d/%d], Training Loss: %.4f, Validation loss: %.4f'
                      %(epoch+1, args.num_epochs, batch_idx+1, total_step, 
                        loss.data[0], val_loss))
                stop = early_stop(val_loss_history)
                if stop:
                    print('=== Early stopping === Validation loss not improving significantly ===')
                    torch.save(decoder.state_dict(),
                               os.path.join(args.model_path, 'decoder-anneal%s-%dcolumns-%d-%d.pkl' % (
                                   args.learning_rate_annealing, df_aligned.shape[1], epoch + 1, batch_idx + 1)))
                    torch.save(encoder.state_dict(),
                               os.path.join(args.model_path, 'encoder-anneal%s-%dcolumns-%d-%d.pkl' % (
                                   args.learning_rate_annealing, df_aligned.shape[1], epoch + 1, batch_idx + 1)))
                    break

            # Save the models
            if (batch_idx+1) % args.save_step == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path, 'decoder-anneal%s-%dcolumns-%d-%d.pkl' % (
                               args.learning_rate_annealing, df_aligned.shape[1], epoch + 1, batch_idx + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path, 'encoder-anneal%s-%dcolumns-%d-%d.pkl' % (
                               args.learning_rate_annealing, df_aligned.shape[1], epoch + 1, batch_idx + 1)))

        if args.learning_rate_annealing:
            adjust_learning_rate(optimizer, epoch+1)

        if stop:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # File paths
    parser.add_argument('--model_path', type=str, default='./TrainedModels',
                        help='path for saving trained models')
    parser.add_argument('--aligned_RRM_path', type=str, default='../data/comineddata_nolinegaps_space_delim.fasta',
                        help='path for aligned RRM input file')
    parser.add_argument('--processed_RRM_path', type=str, default='../data/combined_processed_RRM.csv',
                        help='path for outputting processed aligned_RRM data')
    parser.add_argument('--sep', type=str, default=' ',
                        help='separator for RRM input file, default is space')
    # Preprocessing settings
    parser.add_argument('--preprocessed', action='store_true', default=False,
                        help='if RRM file is preprocessed')
    parser.add_argument('--top_n', type=int, default=82, 
                        help='include top n most populated positions')

    # Control printing/saving
    parser.add_argument('--log_step', type=int , default=20, #10,
                        help='step size for printing log info')
    parser.add_argument('--save_step', type=int , default=500, #1000,
                        help='step size for saving trained models')
    
    # Model hyperparameters
    parser.add_argument('--embed_size', type=int, default=64, #256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--learning_rate_annealing', action='store_true', default=False,
                        help='turn lr annealing on')
    parser.add_argument('--hidden_size', type=int, default=128, #512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=100) #5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()    
    
    # CUDA settings
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
