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

def _compute_loss(pred, label):
    difference = label - pred
    loss = torch.sum( torch.abs(difference))
    return loss, loss.data[0]
    
def evaluate_predictions(dev_data_iter, dev_poss_matches, model, epoch):
    #model.eval()
    losses = []
    for x in dev_data_iter:
        dev_embs, dev_known_labels = Variable(x[0], volatile=True), Variable(x[1], volatile=True)
        prediction = model.forward(dev_embs, dev_poss_matches)
        _, loss = _compute_loss(prediction, dev_known_labels) 
        losses.append(loss)
    
    #model.train()
    return np.mean(losses)
    
def training_loop(
    batch_size, num_epochs, model, optim, data_iter, rna_options, dev_input,
    embed_file = 'unknown_model', 
    print_every=50, eval_every=100):
    epoch = 0
    losses, avg_loss_each_epoch, valid_losses  = [], [], []
     
    total_batches = int(len(data_iter))
    
    #print(rna_options)
    
    poss_matches = Variable(rna_options) # RNA matrix to right multiply 
                
    while epoch <= num_epochs:
        for x in data_iter:
            data, label = Variable(x[0]), Variable(x[1]) 
            
            model.train()
            model.zero_grad()

            prediction = model.forward(data, poss_matches)
            loss_var, loss = _compute_loss(prediction, label) 

            losses.append(loss) 
            loss_var.backward()
            optim.step()

        epoch += 1
        #avg_loss_each_epoch.append(np.mean(losses))
        if epoch % print_every == 0:
            print("TRAIN loss at epoch {0}: {1}".format(epoch, loss ))
            print("TEST loss at epoch {0}: {1}".format(epoch, valid_losses[-1]))
            print('     ')

            #print( "TRAIN avg loss at epoch: ", (epoch), "Avg Loss:", np.mean(losses)/batch_size )            
            #plot_loss(avg_loss_each_epoch, epoch)
        if epoch % eval_every == 0:
            valid_loss = evaluate_predictions(dev_input, poss_matches, model, epoch)
            if epoch > 2000 and valid_loss > valid_losses[-1] :
                print("Lowest TEST loss at epoch {0}: {1}".format(epoch, valid_loss))
                exit() 
            torch.save( model.state_dict(), '%s_%s_SRSavedModel.pth' %(embed_file, valid_loss))                
                
            valid_losses.append(valid_loss)
            

            
            #plot_loss(valid_losses, epoch, False)
    
# new_mat = known_matches - outs
            # loss = torch.bmm(new_mat.view(new_mat.size()[0], 1, new_mat.size()[1]),
            #                    new_mat.view(new_mat.size()[0], new_mat.size()[1], 1)
            #                   )

            #weight_sparsity=torch.sum((model.lin3.weight**2).sum(1))
            #weight_sparsity=0 #torch.sum((torch.abs(model.lin3.weight)).sum(1))
            #loss = torch.sqrt(loss) 
            #loss = torch.div(torch.sum(loss.view(-1)), batch_size) + weight_sparsity
            