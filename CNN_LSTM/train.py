from CNN_LSTM import * 
from util.loader import loader
from cnn_autoencoder import cnn_autoencoder

num_epochs = 2
batch_size = 1
learning_rate = 1e-3

model = cnn_autoencoder()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

latent_representation = []
representation_indices = []
for epoch in range(num_epochs):
    loss  = 0
    for data, label in loader():
        if data.shape[0] != 64:
            continue

        # Lihan has hard time one-hoting integer encoded tensor (He thinks we need both)  
        inp = data % 23
        inp_ = torch.unsqueeze(inp, 2)
        
        data_onehot = torch.FloatTensor(64, 81, 23).zero_()
        data_onehot.scatter_(2, inp_ , 1).float()
        data_onehot = data_onehot.unsqueeze(1)
        data_onehot = data_onehot.transpose(2, 3)
        
        data = Variable(data)
        
        # ===================forward=====================
        encoded, unpool_indices = model.encode(data)

        if epoch == num_epochs-1: # save latent representation here
            rep = encoded.data.squeeze(2).numpy()
            latent_representation.append(rep)
            representation_indices.append(label.numpy())

        output = model.decode( encoded, unpool_indices )      
        loss = criterion(output, Variable(data_onehot))
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))

torch.save(model.state_dict(), './conv_autoencoder.pth')

import pickle 
pickle.dump(
       np.array(latent_representation), open( "./data/cnn_latent_rep.p", "wb" ) )
pickle.dump(
       np.array(representation_indices), open( "./data/cnn_latent_rep_indices.p", "wb" ) )
