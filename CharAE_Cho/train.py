from CharAE_Cho import * 
from util.loader import loader
from autoencoder import CharLevel_autoencoder, cnn_autoencoder
import pickle 

num_epochs = 2
batch_size = 64
learning_rate = 1e-3

criterion = nn.BCEWithLogitsLoss()
model = CharLevel_autoencoder(criterion)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

def train(model, optimizer, num_epochs, batch_size, learning_rate):
    #model.load_state_dict(torch.load('./conv_autoencoder.pth'))
    train_loader, valid_loader = loader()
    
    latent_representation = []
    representation_indices = []
    for epoch in range(num_epochs):
        model.train()

        for index, (data, label) in enumerate(train_loader):
            if data.shape[0] != 64: # this data we offer to the gods 
                continue

            # one-hoting integer encoded tensor ( both needed )  
            inp = data % 22
            inp_ = torch.unsqueeze(inp, 2)
            
            data_onehot = torch.FloatTensor(64, 81, 22).zero_()
            data_onehot.scatter_(2, inp_ , 1).float()
            data_onehot = data_onehot.unsqueeze(1)
            data_onehot = data_onehot.transpose(2, 3)
            #print(data_onehot.shape) #torch.Size([64, 1, 23, 81])
            data = Variable(data)
            
            # ===================forward=====================
            encoder_outputs, encoder_hidden = model.encode(data)
            #print(encoder_outputs.data.shape) # torch.Size([81, 64, 25])

            if epoch == num_epochs-1: # save latent representation here
                rep = encoder_outputs.data.transpose(0,1)
                rep = rep.contiguous().view(64, -1).numpy()
                if index == 100:
                    print('good job, everything looks good: a batchs latent rep has shape', rep.shape)
                latent_representation.append(rep)
                representation_indices.append(label.numpy())
                
            decoder_hidden = encoder_hidden
            #print('deocder input', decoder_input.shape, 'decoder hidden', decoder_hidden.data.shape)
            encoder_outputs = encoder_outputs.transpose(0,1)
            
            loss = model.decode(
                data, data_onehot, decoder_hidden, encoder_outputs,)
            
            if index % 100 == 0:
                print(epoch, index, loss.data[0])
                print( evaluate(model, valid_loader))
                model.train()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # ===================log========================
        torch.save(model.state_dict(), './conv_autoencoder.pth')
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.data[0]))

    pickle.dump(
        latent_representation, open( "./data/latent_rep.p", "wb" ), protocol=4 )
    pickle.dump(
        representation_indices, open( "./data/latent_rep_indices.p", "wb" ), protocol=4)

def evaluate(model, valid_loader ):
    model.eval()
    
    for index, (data, label) in enumerate(valid_loader):
        inp = data % 22
        inp_ = torch.unsqueeze(inp, 2)
            
        data_onehot = torch.FloatTensor(64, 81, 22).zero_()
        data_onehot.scatter_(2, inp_ , 1).float()
        data_onehot = data_onehot.unsqueeze(1)
        data_onehot = data_onehot.transpose(2, 3)
            
        data = Variable(data)
        encoder_outputs, encoder_hidden = model.encode(data)
        decoder_hidden = encoder_hidden
        encoder_outputs = encoder_outputs.transpose(0,1)
        loss = model.decode(data, data_onehot, decoder_hidden, encoder_outputs,)
        return loss.data[0]

if __name__ == '__main__':
    train(model, optimizer, num_epochs, batch_size, learning_rate)