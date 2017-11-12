from CharAE_Cho import *
import torch.nn.functional as F

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size,hidden_size = 320, n_layers=1, dropout_p=0.6,):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.attn = nn.Linear(2*hidden_size, 27)
        self.attn_combine = nn.Linear(2*hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size,)#  bidirectional = True)
        self.out = nn.Linear(hidden_size, 22)

    def forward(self, embedded, hidden, encoder_outputs):
        #print(embedded.data.shape)
        hidden = hidden.view(64, -1, 1)
        #embedded = embedded.unsqueeze(2)
        embedded = embedded.transpose(0,1).transpose(1,2)
        #print( embedded.data.shape, hidden.data.shape,) #torch.Size([64, 25, 1]) 

        all_info = torch.cat((embedded, hidden), 1)
        all_info, embedded, hidden = all_info.squeeze(2), embedded.squeeze(2), hidden.squeeze(2)  
        #print(all_info.data.shape, embedded.data.shape, encoder_outputs.data.shape)
        
        attn_weights = F.softmax( self.attn(all_info) ) # 50 X 81 
        #print(attn_weights.data.shape,  ) #torch.Size([64, 81])
        encoder_outputs = encoder_outputs.transpose(0,1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output) #(64, 25)
    
        #print(output.data.shape, hidden.data.shape)
        for i in range(self.n_layers):
            # inputs of dim (batch, seq_len, input_size)
            output = F.relu(output)
            output, hidden = self.gru(
                output.unsqueeze(0), hidden.unsqueeze(0))

        output = self.out(output)
        return output, hidden, attn_weights


''' simple decoders for debugging purposes 
'''
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size=25, output_size=23, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, output, hidden):
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = (self.out(output))
        
        return output, hidden

class cnn_decoder(nn.Module):
    def __init__(self):
        super(cnn_decoder, self).__init__()

        self.unpool = nn.MaxUnpool1d(9) 
        # makes sense to deconvolve all activations (5 X (64, 5, 1, 81)? ) then 
        # apply linear layer 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(25, 4, 1, stride=1, padding = 0 ), 
            nn.ReLU(True),
        )
        # sigmoid built into BCELogitLoss for faster sigmoiding
        self.linear = (nn.Linear(25, 23))

    def forward(self, x, unpool_indices):
        x = x.squeeze(2)
        x = self.unpool(x, unpool_indices)
        x = x.unsqueeze(2)
        
        # TODO deconvolve and check transposes are needed
        x = x.transpose(1, 3)
        x = (self.linear(x))
        x = x.transpose(2, 3).transpose(1,3)
        return x 
