from CharAE_Cho import *
import torch.nn.functional as F

class rnn_decoder(nn.Module):
    def __init__(self, hidden_size, emit_len):
        super(rnn_decoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.attn = nn.Linear(3*hidden_size, emit_len) # due to bi-directional gru
        self.attn_combine = nn.Linear(3*hidden_size, hidden_size) #TODO changed second argument to 2Xhidden_size
        
        self.gru = nn.GRU(
            input_size = hidden_size,
            hidden_size = 2*hidden_size)
        self.out = nn.Linear(2*hidden_size, 23)
        
    def forward(self, embedded, hidden, encoder_outputs, attention, batch_size):
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        
        if attention:
            all_info = ( torch.cat((embedded, hidden), 1) ) # 64 X (3*64)
            #print(embedded.data.shape, hidden.data.shape )
            attn_weights = F.softmax( self.attn( all_info ), dim =  1 )# 64 X 78 
            
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1) 
            #print(attn_applied.data.shape, attn_weights.data.shape, encoder_outputs.data.shape)
            
            output = torch.cat((embedded, attn_applied), 1)
            output = self.attn_combine(output) 

        else:            
            output = embedded
        output = F.relu(output)
        output, hidden = self.gru(output.unsqueeze(0), hidden.unsqueeze(0))    
        #print(hidden.data.shape, output.data.shape)
        output = self.out(output.squeeze(0))
        return output, hidden, attn_weights

class cnn_decoder(nn.Module): 
    def __init__(self, filter_widths, filter_config, emit_len):
        super(cnn_decoder, self).__init__()

        self.unpool = nn.MaxUnpool1d(3) 
        # deconvolve
        self.filter_widths = filter_widths
        self.filter_config = filter_config 
        self.deconv_layers = []
        for k, filter_num in zip(filter_widths, filter_config):
            padding = k // 2
            if k % 2 == 0:
                padding -= 1

            k_deconv = nn.ConvTranspose2d( filter_num, 1, (23, k),
              stride=1, padding =  (0, padding) )   
            self.deconv_layers.append( k_deconv )

        self.deconv_layers = nn.ModuleList(self.deconv_layers)
        self.linear = nn.Linear( 23* len(filter_config) , 23 )
        # int(np.sum(np.array(filter_config)))
    def forward(self, x, unpool_indices):
        
        #print(x.data.shape, unpool_indices.data.shape) # 64 X 180 X (78//3)
        x = self.unpool(x, unpool_indices)

        #print(x.data.shape) # 64 X 80 X 84
        
        #deconv: each row of activations correspond to a k-width kernel
        # one row is transpose convolved into ( hidden dim = 64 ) X ( seq len = 78 )  
        filter_width_index = 0 # track contiguous rows belonging to current filter width    
        deconvolved = []
        for k, num_filters, deconv_layer in zip(
            self.filter_widths, self.filter_config, self.deconv_layers):
            #print(x.data.shape, filter_width_index, filter_width_index + num_filters)
            activation_to_deconvolve = x[:, filter_width_index:filter_width_index+num_filters,:].unsqueeze(2)  
            #print(activation_to_deconvolve.data.shape)
            activation_to_deconvolve = deconv_layer(activation_to_deconvolve)[:,:,:,:84]
            #print(k, activation_to_deconvolve.data.shape)
            
            deconvolved.append(activation_to_deconvolve)
            filter_width_index += num_filters

        deconvolved = torch.cat(deconvolved, 1)
        #print(deconvolved.data.shape)
        deconvolved = deconvolved.view(-1, 4*23, 84).transpose(1,2)

        x = self.linear( deconvolved ).view(-1, 23)
        return x