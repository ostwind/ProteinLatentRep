import sys
import os
parent_dir = os.path.abspath(__file__ + "/../../")
sys.path.append(parent_dir)

from CharAE_Cho import * 
from util.regularization import MutationNoise, early_stopping 
from autoencoder import CharLevel_autoencoder, hook
from postprocess import extract_latent_rep, _gen_csv
import pickle 
import argparse

parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('--epochs', metavar='N', type=int,
                   default = 20, help='number of epochs')
parser.add_argument('--archi', metavar='M', type = str, default= 'CR_R',
                   help='autoencoder layout')
parser.add_argument('--seq_len', metavar='L', type = int, default= 84,
                   help='autoencoder layout')
parser.add_argument('--alignment', metavar='L', type = str, default= 'aligned',
                   help='aligned, unaligned, delimited')
parser.add_argument('--noise', metavar='L', type = int, default= 1,
                  help='# of evolutionary noise for model to denoise')
args = parser.parse_args()

model_path = './CharAE_Cho/%s_%s_%snoise.pth' %(args.archi, args.alignment, args.noise) #
num_epochs, seq_len = args.epochs, args.seq_len
learning_rate = 0.0002

criterion = nn.CrossEntropyLoss() 
model = CharLevel_autoencoder(criterion, seq_len, layers = args.archi )
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)#, weight_decay=1e-5)
print(args)

def train(model, optimizer, num_epochs):
    model.load_state_dict(torch.load(model_path))
    train_loader, valid_loader = loader(args.alignment)
    validation_loss_history, model_states = [], []
    
    for epoch in range(num_epochs):
        model.train()
        for index, (data, label) in enumerate(train_loader):
            model.batch_size = data.shape[0]  

            noisy_data = Variable(  MutationNoise( data ) )  
            data  = Variable(data)
        
            embedded = model.Embed_encode(noisy_data)

            # ===================forward=====================
            if args.archi == 'C_C':
                activations, unpool_indices = model.Conv_encode(embedded)
                loss = model.Conv_decode(data, activations, unpool_indices)

            else:
                if args.archi == 'R_R':
                    encoder_outputs, encoder_hidden = model.GRU_encode(embedded)
                else:
                    encoder_outputs, encoder_hidden = model.NMT_encode(embedded)
                #print(encoder_hidden.data.shape)
                #encoder_hidden = encoder_outputs[:,-1,:] 

                loss = model.GRU_decode(data, encoder_hidden, encoder_outputs)
            
                #l1_reg = encoder_hidden.transpose(
                #    0,1).contiguous().view(data.shape[0], -1).norm(p = 1, dim = 1) 
                # l1_reg = encoder_outputs.contiguous().view(data.shape[0], -1).norm(p = 1, dim = 1) 
                #print(l1_reg.data.shape)
                #loss += 0.01*l1_reg.sum()
                
            if index % 200 == 0:
                print('%d, %d, %.3f' %(
                    epoch, index, loss.data[0] ) )
                valid_loss = evaluate(model, valid_loader) 
                print('validation loss: %.3f' %valid_loss )
                
                #validation_loss_history.append(valid_loss)
                model.train()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            # exploding gradient happening in GRU, clip gradient here
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm = 5)
            optimizer.step()

        # ===================log========================
        torch.save(model.state_dict(), model_path)
        #print('epoch [{}/{}], loss:{:.4f}'
        #    .format(epoch+1, num_epochs, loss.data[0]))
        print('________________ epoch ', epoch, ' _______________' )
    
    #torch.save(model_states[-1], model_path)

def evaluate(model, valid_loader):
    model.eval()
    for index, (data, label) in enumerate(valid_loader):
        model.batch_size = data.shape[0]
        data = Variable(data, volatile = True)
        embedded = model.Embed_encode(data)     
 
        if args.archi == 'C_C':
            activations, unpool_indices = model.Conv_encode(embedded)
            loss = model.Conv_decode(data, activations, unpool_indices)

        else:
            if args.archi == 'R_R':
                encoder_outputs, encoder_hidden = model.GRU_encode(embedded, inference = True)    
            else:        
                encoder_outputs, encoder_hidden = model.NMT_encode(embedded, inference = True)
            #encoder_hidden = encoder_outputs[:, -1, :]

            loss = model.GRU_decode( data, encoder_hidden, encoder_outputs, data)
            #l1_reg = encoder_hidden.transpose(
            #        0,1).contiguous().view(64, -1).norm(p = 1, dim = 1) 
            #l1_reg = encoder_outputs.contiguous().view(data.shape[0], -1).norm(p = 1, dim = 1)  
            #loss += 0.01*l1_reg.sum()
                
        return loss.data[0]

if __name__ == '__main__':
    train(model, optimizer, num_epochs)
    print('____________________extracting latent rep___________________')
    rep_path, indices_path = extract_latent_rep(model, args, model_path)
    _gen_csv( args )
    