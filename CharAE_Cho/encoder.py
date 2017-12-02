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
        input = input.contiguous().view(1, 64, self.hidden_size)#input.transpose(0,2).transpose(2,1)
        #print(input.data.shape, hidden.data.shape)
        output = input
        for i in range(self.n_layers):
            output, hidden = self.gru(output, 
            hidden)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(2, 64, self.hidden_size))

class cnn_encoder(nn.Module):
    def __init__(self, filter_widths, num_filters_per_width, char_embedding_dim):#, seq_len):
        super(cnn_encoder, self).__init__()
        self.filter_size_range = filter_widths
        self.char_embedding_dim = char_embedding_dim
        self.num_filters_per_width = num_filters_per_width  #possible sizes * num_filters = decoder hidden size

        # 1 conv layer with dynamic padding 
        # this enable kernels with varying widths a la Cho's NMT (2017) 
        self.filter_banks = [] 
        for k in self.filter_size_range:
            padding = k //2 
            
            self.k_filters = nn.Conv2d(
                    1, self.num_filters_per_width, 
                    # for kernel and padding the dimensions are (H, W)
                    (self.char_embedding_dim, k), padding=(0, padding), 
                    stride=1)  
            self.filter_banks.append( self.k_filters )

        self.pool = nn.MaxPool1d( 3 , return_indices=True)  

    def forward(self, x, seq_len, collect_filters):
        all_activations = []
        all_unpool_indices = []
        for k, k_sized_filters in zip(
            self.filter_size_range, self.filter_banks): 
            #print(k_sized_filters.weight)

            activations = F.relu(k_sized_filters(x))    
            
            if k % 2 == 0: # even kernel widths: skip last position 
                input_indices = torch.LongTensor(range(seq_len))
                activations = activations.index_select( 3, Variable(input_indices)) 
    
            #print('convolved width %s kernels for activations in shape %s' %(k, activations.data.shape) )
            
            activations = activations.squeeze(2)
            activations, unpool_indices = self.pool(activations)
            activations = activations.unsqueeze(2)
            all_unpool_indices.append(unpool_indices)
            all_activations.append(activations)

            #f_weight = k_sized_filters.weight.data.squeeze(1).numpy()
            #f_bias = k_sized_filters.bias.data.numpy()
            # state_dict() dosen't see all kernel widths, only largest kernels with largest width
            # if collect_filters and k == 3:                
            #     pickle.dump( f_weight, open( "%s.p" %(k), "wb" ) )
            #     pickle.dump( f_bias, open( "%sb.p" %(k), "wb" ) )

        activation_tensor = torch.cat(all_activations, 1)
        all_unpool_indices = torch.cat(all_unpool_indices, 1)
        # if collect_filters:
        #     activations_dir = './CharAE_Cho/activations/'
        #     for i in range(2000):
        #         if os.path.isfile(activations_dir + '%s.p' %(i)):
        #             continue
        #         pickle.dump( activation_tensor.data.squeeze(2).numpy(), 
        #         open( activations_dir + "%s.p" %(i), "wb" ) )
        #         pickle.dump( all_unpool_indices.data.squeeze(2).numpy(), 
        #         open( activations_dir + "%spool.p" %(i), "wb" ) )
        #         break
        return activation_tensor#, all_unpool_indices 