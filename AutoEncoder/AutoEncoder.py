"""Autoencoder for learning latent representation
Refs:
SparseAutoencoder: 
    https://discuss.pytorch.org/t/how-to-create-a-sparse-autoencoder-neural-network-with-pytorch/3703
VAE:
    https://github.com/pytorch/examples/blob/master/vae/main.py"""

import numpy as np
import pandas as pd
import torch
import os
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim


parser = argparse.ArgumentParser(description='VAE for RRM')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', 
    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument('--save', type=str,  default='model.pt',
    help='path to save the final model')


class RRM_Dataset(Dataset):

    """One hot encoded RRM dataset"""

    def __init__(self, indices, root_dir='../data', transform=None):
        """
        Args:
            csv_file (string): directory of csv file with all RRM sequences.
            root_dir (string): directory with all the csv file for
            individual RRM csv.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(RRM_Dataset).__init__()
        self.root_dir = root_dir
        self.names = indices
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        rrm_name = os.path.join(self.root_dir, self.names[idx])
        rrm = pd.read_csv(rrm_name+'.csv', index_col=0).as_matrix().astype('float')
        rrm = torch.from_numpy(rrm).contiguous().float()
        sample = {'name': rrm_name, 'seq': rrm}

        if self.transform:
            sample = self.transform(sample)

        return sample




class VAE(nn.Module):

    """largely taken from:
    https://github.com/pytorch/examples/blob/master/vae/main.py"""
    def __init__(self):
        super(VAE, self).__init__()

        self.lin1 = nn.Linear(3608, 1800)
        self.lin2 = nn.Linear(1800, 900)
        self.lin31 = nn.Linear(900, 100)
        self.lin32 = nn.Linear(900, 100)

        self.lin4 = nn.Linear(100, 900)
        self.lin5 = nn.Linear(900, 1800)
        self.lin6 = nn.Linear(1800, 3608)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.lin1(x)) #1800
        h2 = self.relu(self.lin2(h1)) #900
        return self.lin31(h2), self.lin32(h2) #100, 100

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4 = self.relu(self.lin4(z)) #900
        h5 = self.relu(self.lin5(h4)) #1800
        return self.sigmoid(self.lin6(h5)) #3608

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):

    """ref: https://arxiv.org/abs/1312.6114"""

    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD/=args.batch_size*3608

    return KLD + BCE


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, dic in enumerate(train_loader):
        data = Variable(dic['seq'].view(-1, 3608))
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        z_batch, recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
        pd.DataFrame(data.data.numpy()).to_csv('original_epoch%d' % epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
      epoch, train_loss / len(train_loader.dataset)))



def test(epoch):
    model.eval()
    test_loss = 0
    for i, dic in enumerate(val_loader):
        data = Variable(dic['seq'].view(-1, 3608), volatile=True)
        if args.cuda:
            data = data.cuda()
        z_batch, recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data,
            mu, logvar).data[0]
    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    pd.DataFrame(recon_batch.data.numpy()).to_csv('reconstructed_epoch%d'%epoch)
    pd.DataFrame(data.data.numpy()).to_csv('original_epoch%d'%epoch)
    return (test_loss)




if __name__ == "__main__":

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_index = pd.read_csv('../data/train_index.csv',header=None).iloc[:,0]
    val_index = pd.read_csv('../data/val_index.csv',header=None).iloc[:,0]
    train_loader = RRM_Dataset(indices=train_index)
    train_loader = DataLoader(train_loader, batch_size=args.batch_size,shuffle=True)
    val_loader = RRM_Dataset(indices=val_index)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=True)

    model = VAE()
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = [0, np.infty]
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = test(epoch)
        if loss < best_loss[1]:
            best_loss[0] = epoch
            best_loss[1] = loss
        with open(args.save, 'wb') as f:
            torch.save(model.state_dict(), args.save)









