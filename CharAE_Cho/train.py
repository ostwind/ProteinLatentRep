import sys
import os
parent_dir = os.path.abspath(__file__ + "/../../")
sys.path.append(parent_dir)

from CharAE_Cho import * 
from util.regularization import _add_swap_noise, early_stopping 
from autoencoder import CharLevel_autoencoder
from postprocess import extract_latent_rep
import pickle 
import argparse

parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('--epochs', metavar='N', type=int,
                   default = 7, help='number of epochs')
parser.add_argument('--archi', metavar='M', type = str, default= 'CR_R',
                   help='autoencoder layout')
parser.add_argument('--seq_len', metavar='L', type = int, default= 78,
                   help='autoencoder layout')
parser.add_argument('--swap_noise', metavar='L', type = int, default= 0,
                   help='number of symbol swaps for model to denoise')
args = parser.parse_args()

model_path = './C_C.pth'
num_epochs, seq_len = args.epochs, args.seq_len
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
model = CharLevel_autoencoder(criterion, seq_len, layers = args.archi )
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=1e-5)

def train(model, optimizer, num_epochs):
    model.load_state_dict(torch.load(model_path))
    train_loader, valid_loader = loader()
    validation_loss_history, model_states = [], []
    
    for epoch in range(num_epochs):
        model.train()
        for index, (data, label) in enumerate(train_loader):
            data = Variable(data)
            embedded = model.Embed_encode(data)
            # ===================forward=====================
            if args.archi == 'C_C':
                activations, unpool_indices = model.Conv_encode(embedded)
                loss = model.Conv_decode(data, activations, unpool_indices)

            else:
                if args.archi == 'R_R':
                    encoder_outputs, encoder_hidden = model.GRU_encode(embedded)
                else:
                    encoder_outputs, encoder_hidden = model.NMT_encode(embedded)
                loss = model.GRU_decode(data, encoder_hidden, encoder_outputs)
            
            if index % 100 == 0:
                print(epoch, index, loss.data[0])
                valid_loss = evaluate(model, valid_loader)
                print('validation loss: ', valid_loss )

                validation_loss_history.append(valid_loss)
                steps_considered = 5
                model_states.append( model.state_dict() )
                if len(model_states) > steps_considered + 1:
                    del model_states[0]

                if early_stopping(validation_loss_history, steps_considered):
                    torch.save( model_states[0], model_path)
                    return 
    
                model.train()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================log========================
        torch.save(model.state_dict(), model_path)
        #print('epoch [{}/{}], loss:{:.4f}'
        #    .format(epoch+1, num_epochs, loss.data[0]))
        print('________________ epoch ', epoch, ' _______________' )
    
    torch.save(model_states[-1], model_path)

def evaluate(model, valid_loader):
    model.eval()
    for index, (data, label) in enumerate(valid_loader):
        data = Variable(data, volatile = True)
        embedded = model.Embed_encode(data)     
 
        if args.archi == 'C_C':
            activations, unpool_indices = model.Conv_encode(embedded)
            loss = model.Conv_decode(data, activations, unpool_indices)

        else:
            if args.archi == 'R_R':
                encoder_outputs, encoder_hidden = model.GRU_encode(embedded)    
            else:        
                encoder_outputs, encoder_hidden = model.NMT_encode(embedded)
            loss = model.GRU_decode( data, encoder_hidden, encoder_outputs)
        return loss.data[0]

if __name__ == '__main__':
    train(model, optimizer, num_epochs)
    print('____________________extracting latent rep___________________')
    rep_path, indices_path = extract_latent_rep(model, args, model_path)
    #filter(rep_path, indices_path)

    