import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO


def training_loop(batch_size, num_epochs, model, optim, data_iter, rna_options):
    step = 0
    epoch = 0
    losses = []
    total_batches = int(len(data_iter) / batch_size)
    poss_matches = Variable(rna_options)
    while epoch <= num_epochs:
        for x in data_iter:
            learned_embs = Variable(x[0])
            known_matches = Variable(x[1])
            
            model.train()
            model.zero_grad()

            outs = test_model.forward(3, learned_embs, poss_matches)
            # print(outs.size())
            new_mat = known_matches - outs
            loss = torch.bmm(new_mat.view(new_mat.size()[0], 1, new_mat.size()[1]),
                               new_mat.view(new_mat.size()[0], new_mat.size()[1], 1)
                              )
            loss = torch.sqrt(loss)
            loss = torch.div(torch.sum(loss.view(-1)), batch_size)
            losses.append(loss[0].data.numpy()) 
            loss.backward()
            optim.step()


        epoch += 1
        if epoch % 500 == 0:
            print( "Epoch:", (epoch), "Avg Loss:", np.mean(losses)/(total_batches*epoch) )
        
        step += 1

