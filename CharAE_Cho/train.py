from CharAE_Cho import * 
from util.loader import loader
from autoencoder import CharLevel_autoencoder
import pickle 

num_epochs = 7
batch_size = 64
learning_rate = 1e-3
max_batch_len = 78

criterion = nn.CrossEntropyLoss()
model = CharLevel_autoencoder(criterion)#, seq_len)
#TODO: weight decay boosted from 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps = 0.0001,
                             weight_decay=1e-5)

# def _add_swap_noise(batch, seq_len ):
#     noisy_batch = batch.clone()
#     sample_len, num_samples = batch.shape[1], batch.shape[0]
#     #total_swaps = sample_len//40 #as per cho, see https://arxiv.org/pdf/1710.11041.pdf @denoising
#     total_swaps = 3
    
#     for a_swap in range(total_swaps):
#         swap_index = np.random.randint( sample_len, size = 1)[0]
#         while swap_index >= sample_len -1:
#             swap_index = np.random.randint( sample_len,size = 1)[0]
        
#         #print(swap_index, np.vstack([noisy_batch[0, swap_index+1], noisy_batch[0, swap_index]] ) )
#         value_cache = noisy_batch[:, swap_index].clone()
#         noisy_batch[:, swap_index] = noisy_batch[:, swap_index+1]
#         noisy_batch[:, swap_index+1] = value_cache
#         #print(swap_index, np.vstack([noisy_batch[0, swap_index+1], noisy_batch[0, swap_index]] ) )
        
#     return noisy_batch

def train(model, optimizer, num_epochs, batch_size, learning_rate):
    model.load_state_dict(torch.load('./autoencoder.pth'))
    train_loader, valid_loader = loader()
    
    latent_representation = []
    representation_indices = []
    for epoch in range(num_epochs):
        model.train()
        for index, (data, label) in enumerate(train_loader):
            #noisy_data = _add_swap_noise(data, max_batch_len)
            noisy_data = Variable(data)#, Variable(data)
            
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
                
            #print('deocder input', decoder_input.shape, 'decoder hidden', decoder_hidden.data.shape)
            
            #send in corrupted data to recover clean data
            loss = model.decode(
                data, encoder_hidden, encoder_outputs, max_batch_len)
            
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
        encoder_outputs, encoder_hidden = model.encode(Variable(data, volatile=True), max_batch_len)
        loss = model.decode( data, encoder_hidden, encoder_outputs, max_batch_len, index)
        return loss.data[0]

if __name__ == '__main__':
    train(model, optimizer, num_epochs, batch_size, learning_rate)


    