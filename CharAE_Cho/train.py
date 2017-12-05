from CharAE_Cho import * 
from util.loader import loader
from autoencoder import CharLevel_autoencoder
import pickle 

num_epochs = 3
batch_size = 64
learning_rate = 1e-3
max_batch_len = 84

criterion = nn.BCEWithLogitsLoss()
model = CharLevel_autoencoder(criterion)#, seq_len)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

#SEQ format:
#trim padding to longest seq in batch 
#one hot <- predict this
#add noise <- input into GRU

def _one_hot(batch, seq_len):
        inp = batch % 22
        inp_ = torch.unsqueeze(inp, 2)
        
        batch_onehot = torch.FloatTensor(64, seq_len, 22).zero_()
        batch_onehot.scatter_(2, inp_ , 1).float()
        batch_onehot = batch_onehot.transpose(1,2)
        # (batch_size, 1, num_symbols, seq_len)
        return batch_onehot

def _add_swap_noise(batch, seq_len ):
    noisy_batch = batch.clone()
    sample_len, num_samples = batch.shape[1], batch.shape[0]
    #total_swaps = sample_len//40 #as per cho, see https://arxiv.org/pdf/1710.11041.pdf @denoising
    total_swaps = 5
    
    for a_swap in range(total_swaps):
        swap_index = np.random.randint( sample_len, size = 1)[0]
        while swap_index >= sample_len -1:
            swap_index = np.random.randint( sample_len,size = 1)[0]
        
        #print(swap_index, np.vstack([noisy_batch[0, swap_index+1], noisy_batch[0, swap_index]] ) )
        value_cache = noisy_batch[:, swap_index].clone()
        noisy_batch[:, swap_index] = noisy_batch[:, swap_index+1]
        noisy_batch[:, swap_index+1] = value_cache
        #print(swap_index, np.vstack([noisy_batch[0, swap_index+1], noisy_batch[0, swap_index]] ) )
        
    return noisy_batch

def train(model, optimizer, num_epochs, batch_size, learning_rate):
    model.load_state_dict(torch.load('./autoencoder.pth'))
    train_loader, valid_loader = loader()
    
    latent_representation = []
    representation_indices = []
    for epoch in range(num_epochs):
        model.train()
        for index, (data, label) in enumerate(train_loader):
            batch_onehot = (_one_hot(data, max_batch_len))
            noisy_data = _add_swap_noise(data, max_batch_len)
            noisy_data = Variable(noisy_data)#, Variable(data)
            
            # ===================forward=====================
            if epoch != num_epochs - 1:
                encoder_outputs, encoder_hidden = model.encode(noisy_data, max_batch_len)
            #print(encoder_outputs.data.shape, encoder_hidden.data.shape) 
            else:
                encoder_outputs, encoder_hidden = model.encode(
                    noisy_data, max_batch_len)#, collect_filters = True)
                rep = encoder_outputs.data.view(64, -1).numpy()
                if index == 100:
                    print('good job, everything looks good: a batchs latent rep has shape', rep.shape)
                latent_representation.append(rep)
                representation_indices.append(label.numpy())
                
            decoder_hidden = encoder_hidden
            #print('deocder input', decoder_input.shape, 'decoder hidden', decoder_hidden.data.shape)
            
            #send in corrupted data to recover clean data
            loss = model.decode(
                noisy_data, batch_onehot, decoder_hidden, encoder_outputs, max_batch_len)
            
            if index % 100 == 0:
                print(epoch, index, loss.data[0])
                print( evaluate(model, valid_loader))
                model.train()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # ===================log========================
        torch.save(model.state_dict(), './autoencoder.pth')
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.data[0]))

    pickle.dump(
        latent_representation, open( "./data/latent_rep.p", "wb" ), protocol=4 )
    pickle.dump(
        representation_indices, open( "./data/latent_rep_indices.p", "wb" ), protocol=4)

def evaluate(model, valid_loader ):
    model.eval()
    for index, (data, label) in enumerate(valid_loader):
        batch_onehot = _one_hot(data, max_batch_len)
        data = Variable(data)

        encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
        decoder_hidden = encoder_hidden
        
        loss = model.decode(data, batch_onehot, decoder_hidden, encoder_outputs, max_batch_len)
        return loss.data[0]

if __name__ == '__main__':
    train(model, optimizer, num_epochs, batch_size, learning_rate)


    def _trim_padding(batch):    
        # mask batch to detect non-zero elements than sum along the sequence length axis
        batch_max_len = torch.max(batch.gt(0).cumsum(dim = 1))
        return batch[:, :batch_max_len], batch_max_len 


    