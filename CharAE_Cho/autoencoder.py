from CharAE_Cho import *
from encoder import cnn_encoder, rnn_encoder
from decoder import AttnDecoderRNN

class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion):#, seq_len):
            super(CharLevel_autoencoder, self).__init__()
            self.char_embedding_dim = 64
            self.filter_widths = list(range(1, 7)) 
            self.num_filters_per_width = 20 # too wide leads to memorization
            # TODO experiment with more conv layers
            self.encoder_embedding = nn.Embedding(22, self.char_embedding_dim)
            self.cnn_encoder = cnn_encoder(
            filter_widths = self.filter_widths,
            num_filters_per_width = self.num_filters_per_width,
            char_embedding_dim = self.char_embedding_dim)
            #seq_len = self.seq_len)
            
            self.decoder_hidden_size = len(self.filter_widths) * self.num_filters_per_width
            self.rnn_encoder = rnn_encoder( 
            hidden_size = self.decoder_hidden_size )

            #self.rnn_emits_len = 1
            self.decoder_embedding = nn.Embedding(22, self.decoder_hidden_size)
            self.attention_decoder = AttnDecoderRNN(
                  hidden_size = self.decoder_hidden_size, output_size = 84//3)
            self.criterion = criterion

      def encode(self, data, seq_len, collect_filters = False):
            encoder_embedded = self.encoder_embedding(data).unsqueeze(1).transpose(2,3) 
            encoded = self.cnn_encoder.forward(encoder_embedded, seq_len, collect_filters)
            encoded = encoded.squeeze(2)
      
            encoder_hidden = self.rnn_encoder.initHidden()
            #print('encoded dimensions', encoded.data.shape, 'encoder_hidden', self.encoder_hidden.data.shape)
            #encoded dimensions torch.Size([64, 25, 27]) encoder_hidden torch.Size([1, 64, 25])
            
            # 2 times hidden size for bi-directional gru 
            encoder_outputs = Variable(torch.zeros(64, seq_len//3, 2*self.decoder_hidden_size))
            for symbol_ind in range(seq_len//3):#self.rnn_emits_len): 
                  output, encoder_hidden = self.rnn_encoder.forward(
                        encoded[:,:,symbol_ind], encoder_hidden)
                  #print(output.data.shape) # (81, 64, 128)
                  encoder_outputs[:, symbol_ind,:] = output[0]
            return encoder_outputs, encoder_hidden

      def decode(self, noisy_data, target_data, encoder_hidden, encoder_outputs, seq_len):   
            loss = 0
            decoder_hidden = encoder_hidden
            #print(target_data.data.shape)
            for amino_acid_index in range(seq_len): 
                  target_amino_acid = target_data[ :, :, amino_acid_index]#.long()
                  decoder_input = noisy_data.data[:, amino_acid_index].unsqueeze(1)#.transpose(0,1)    
                  decoder_embedded = self.decoder_embedding(decoder_input)
                 
                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_output, decoder_hidden, attn_weights = self.attention_decoder.forward(
                  decoder_embedded, decoder_hidden, encoder_outputs, seq_len//3)
                  #print(decoder_output.data.shape, target_amino_acid.data.shape)   # torch.Size([64, 23])
                  
                  loss += self.criterion(
                        decoder_output,
                        Variable(target_amino_acid) ) 
            return loss 

# preliminary model
# class cnn_autoencoder(nn.Module):
#       def __init__(self):
#             super(cnn_autoencoder, self).__init__()
#             self.encoder = cnn_encoder()
#             self.decoder = cnn_decoder()
#             self.embedding = nn.Embedding(22, 4)
            
#       def encode(self, data):
#             char_embeddings = self.embedding(data).unsqueeze(1).transpose(2,3) 
#             encoded, unpool_indices = self.encoder.forward(char_embeddings)
#             return encoded, unpool_indices

#       def decode(self, data, unpool_indices):
#             reconstructed = self.decoder.forward(data, unpool_indices)
#             return reconstructed