from CharAE_Cho import * 

class rnn_encoder(nn.Module):
    def __init__(self, input_size = 81, hidden_size = 320, n_layers=1):
        super(rnn_encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #print(input.data.shape)
        # input: (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE) hidden: (1, BATCH_SIZE, HIDDEN_SIZE)
        input = input.contiguous().view(1, 64, self.hidden_size)#input.transpose(0,2).transpose(2,1)
        output = input
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, 64, self.hidden_size))
        
class cnn_encoder(nn.Module):
    def __init__(self):
        super(cnn_encoder, self).__init__()
        self.filter_size_range = list(range(2, 10))
        self.char_embedding_dim = 128
        self.num_w_wide_filters = 40  #8*40 = 320
        # 1 conv layer with dynamic padding 
        # this enable kernels with varying widths a la Cho's NMT (2017) 
        self.filter_banks = [] 
        for k in self.filter_size_range:
            padding = k //2 
            
            self.k_filters = nn.Sequential(
                nn.Conv2d(
                    1, self.num_w_wide_filters, 
                    # for kernel and padding the dimensions are (H, W)
                    (self.char_embedding_dim, k), padding=(0, padding), 
                    stride=1),  
                nn.ReLU(True),
            )
            self.filter_banks.append( self.k_filters )

        # decoder depools 
        self.pool = nn.MaxPool1d( 3 , return_indices=True)  

    def forward(self, x):
        all_activations = []
        all_unpool_indices = []
        for k, k_sized_filters in zip(
            self.filter_size_range, self.filter_banks): 
            activations = k_sized_filters(x)    
            
            if k % 2 == 0:
                input_indices = torch.LongTensor(range(81))
                activations = activations.index_select( 3, Variable(input_indices)) 

            #print('convolved width %s kernels for activations in shape %s' %(k, activations.data.shape) )
            
            activations = activations.squeeze(2)
            activations, unpool_indices = self.pool(activations)
            activations = activations.unsqueeze(2)
            #all_unpool_indices.append(unpool_indices)

            all_activations.append(activations)
        activation_tensor = torch.cat(all_activations, 1)
        #all_unpool_indices = torch.cat(all_unpool_indices, 1)

        return activation_tensor#, all_unpool_indices 