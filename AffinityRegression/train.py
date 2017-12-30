import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from Bio import SeqIO
import matplotlib.pyplot as plt

def plot_loss(loss_list, epoch, train=True):
    plt.figure(figsize=(8,12))
    plt.plot(range(len(loss_list)), loss_list, label="last loss value: {0}".format(loss_list[-1]))
    if train:
        plt.title("Train Loss Curve ({} Epochs)".format(epoch))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("Train_Loss_Curve.png")
    else:
        plt.title("Valid Loss Curve ({} Epochs)".format(epoch))        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("Validation_Loss_Curve.png")
    plt.close()

    
def evaluate_predictions(dev_data_iter, dev_poss_matches, model, batch_size, epoch):
    model.eval()
    losses = []
    for x in dev_data_iter:
        dev_embs, dev_known_labels = Variable(x[0]), Variable(x[1])
        outs = model.forward(1, dev_embs, dev_poss_matches)
        new_mat = dev_known_labels - outs
        loss = torch.bmm(new_mat.view(new_mat.size()[0], 1, new_mat.size()[1]),
                           new_mat.view(new_mat.size()[0], new_mat.size()[1], 1)
                          )
        loss = torch.sqrt(loss)
        loss = torch.div(torch.sum(loss.view(-1)), batch_size)
        losses.append(loss[0].data.numpy())
    print("Average loss on dev set at epoch {0}: {1}".format(epoch, np.mean(losses)))
    model.train()
    return np.mean(losses)
    
    

    
    
    
def training_loop(batch_size, num_epochs, model, optim, data_iter, rna_options, dev_input, print_every=50, eval_every=100):
    step = 0
    epoch = 0
    losses = []
    avg_loss_each_epoch = []
    valid_losses = []
    total_batches = int(len(data_iter))
    poss_matches = Variable(rna_options)
    while epoch <= num_epochs:
        for x in data_iter:
            learned_embs = Variable(x[0])
            known_matches = Variable(x[1])
            
            model.train()
            model.zero_grad()

            outs = model.forward(3, learned_embs, poss_matches)
            new_mat = known_matches - outs
            loss = torch.bmm(new_mat.view(new_mat.size()[0], 1, new_mat.size()[1]),
                               new_mat.view(new_mat.size()[0], new_mat.size()[1], 1)
                              )

            #weight_sparsity=torch.sum((model.lin3.weight**2).sum(1))
            weight_sparsity=torch.sum((torch.abs(model.lin3.weight)).sum(1))
            loss = torch.sqrt(loss) 
            loss = torch.div(torch.sum(loss.view(-1)), batch_size) + weight_sparsity
            losses.append(loss[0].data.numpy()) 
            loss.backward()
            optim.step()


        epoch += 1
        avg_loss_each_epoch.append(np.mean(losses))
        if epoch % print_every == 0:
            print( "Epoch:", (epoch), "Avg Loss:", np.mean(losses)/(total_batches*epoch) )            
            #plot_loss(avg_loss_each_epoch, epoch)
        if epoch % eval_every == 0:
            valid_losses.append(evaluate_predictions(dev_input, poss_matches, model, batch_size, epoch))
            #plot_loss(valid_losses, epoch, False)
        step += 1    
    

