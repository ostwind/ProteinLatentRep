from CharAE_Cho import * 
import torch.nn.functional as F
import pickle 

class rnn_encoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(rnn_encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
    def forward(self, input, hidden):
        # input: (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE) hidden: (1, BATCH_SIZE, HIDDEN_SIZE)
        #print(input.data.shape, hidden.data.shape)

        output = input.contiguous().unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hidden_size))

class cnn_encoder(nn.Module):
    def __init__(self, filter_widths, filter_config, char_embedding_dim):#, seq_len):
        super(cnn_encoder, self).__init__()
        self.filter_size_range = filter_widths
        self.char_embedding_dim = char_embedding_dim
        #self.num_filters_per_width = num_filters_per_width  #possible sizes * num_filters = decoder hidden size

        # 1 conv layer with dynamic padding 
        # this enable kernels with varying widths a la Cho's NMT (2017) 
        self.filter_banks = [] 
        for k, cur_num in zip(filter_widths, filter_config):
            padding = k //2 
            
            self.k_filters = nn.Conv2d(
                    1, cur_num, 
                    # for kernel and padding the dimensions are (H, W)
                    (self.char_embedding_dim, k), padding=(0, padding), 
                    stride=1)  
            self.filter_banks.append( self.k_filters )
        self.filter_banks = nn.ModuleList(self.filter_banks)

        self.pool = nn.MaxPool1d( 6 , return_indices=True)  

    def forward(self, x, seq_len):
        all_activations, all_unpool_indices = [], []
        for k, k_sized_filters in zip(self.filter_size_range, self.filter_banks): 
            #print(k_sized_filters.weight)
            activations = F.selu(k_sized_filters(x))    
            
            if k % 2 == 0: # even kernel widths: skip last position 
                activations = activations[:, :, :, :seq_len] # batch size, num kernels, 1, seq_len
                
            #print('convolved width %s kernels for activations in shape %s' %(k, activations.data.shape) )
            
            activations = activations.squeeze(2)
            activations, unpool_indices = self.pool(activations)
            activations = activations.unsqueeze(2)
            all_unpool_indices.append(unpool_indices)
            all_activations.append(activations)

        activation_tensor = torch.cat(all_activations, 1)
        all_unpool_indices = torch.cat(all_unpool_indices, 1)
        
        return activation_tensor, all_unpool_indices 