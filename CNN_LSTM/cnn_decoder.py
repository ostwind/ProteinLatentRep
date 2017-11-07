from CNN_LSTM import *
import torch.nn.functional as F

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
        #print(x.data.shape)
        return x 