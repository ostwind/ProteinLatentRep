from CharAE_Cho import *
from encoder import cnn_encoder, rnn_encoder
from decoder import rnn_decoder, cnn_decoder
import random 

class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion, seq_len, layers ):
            super(CharLevel_autoencoder, self).__init__()
            
            self.layers = layers
            self.seq_len = seq_len
            self.emit_len = seq_len//6 #CR|R C|C conv encoder -> pooled activations 
            if layers == 'R_R': 
                  self.emit_len = seq_len

            self.layers = layers    
            self.char_embedding_dim = 16
            self.encoder_embedding = nn.Embedding(23, self.char_embedding_dim)
            
            self.filter_widths = list(range(1, 8))
            #self.filter_config = [10, 10, 20, 20, 10, 10, 10] 
            self.filter_config = [20, 20, 40, 40, 20, 20, 20] # too wide leads to memorization
            self.cnn_encoder = cnn_encoder(
            filter_widths = self.filter_widths,
            filter_config = self.filter_config,
            char_embedding_dim = self.char_embedding_dim)
            
            self.decoder_hidden_size = int(np.sum(np.array(self.filter_config)) )
            if layers == 'R_R':
                  self.decoder_hidden_size = self.char_embedding_dim
            self.rnn_encoder = rnn_encoder(
                  hidden_size = self.decoder_hidden_size )

            # _____________________________ decoder ________________________________
            self.decoder_embedding = nn.Embedding(23, self.decoder_hidden_size)
            self.attention_decoder = rnn_decoder(
                  hidden_size = self.decoder_hidden_size, emit_len = self.emit_len)
            self.criterion = criterion

            self.deconv_decoder = cnn_decoder( 
            filter_widths = self.filter_widths,
            filter_config = self.filter_config,
            emit_len = self.emit_len )

      def Embed_encode(self, data):
            encoder_embedded = self.encoder_embedding(data)
            embedded = encoder_embedded.unsqueeze(1).transpose(2,3) 
            return embedded

      def Conv_encode(self, x):
            activations, unpool_indices = self.cnn_encoder.forward(x, self.seq_len)
            return activations.squeeze(2), unpool_indices

      def GRU_encode(self, activations):
            if self.layers == 'R_R':
                  activations = activations.transpose(1,3).contiguous().view(64, self.emit_len, -1)
            
            else: 
                  activations = activations.transpose(1, 2).contiguous()

            encoder_hidden = self.rnn_encoder.initHidden()
            encoder_outputs = Variable(torch.zeros(64, self.emit_len, 2*self.decoder_hidden_size))
            
            for symbol_ind in range(self.emit_len): 
                  current_symbol = activations[:,symbol_ind,:] 
                  output, encoder_hidden = self.rnn_encoder.forward(
                        current_symbol, encoder_hidden)
                  encoder_outputs[:, symbol_ind,:] = output[0]

            return encoder_outputs, encoder_hidden 

      def NMT_encode(self, embedded ):
            #if self.layers != 'R|R':
            activations, _ = self.Conv_encode(embedded)
                  #print(activations.data.shape) #(64, 360, 26)
                  #encoder_outputs, encoder_hidden = self._biGRU_layer(activations)
            encoder_outputs, encoder_hidden = self.GRU_encode(activations)

            return encoder_outputs, encoder_hidden

      def GRU_decode(self, target_data, encoder_hidden, encoder_outputs, attention = True):   
            decoder_hidden = encoder_hidden
            input_embedded = Variable(torch.LongTensor([17]).repeat(64)) # SOS token
            input_embedded = self.decoder_embedding( input_embedded )
            
            sequence_loss = 0
            for symbol_index in range(self.seq_len): 
                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_output, decoder_hidden = self.attention_decoder.forward(
                  input_embedded, decoder_hidden, encoder_outputs, attention)
                  values, input_symbol = decoder_output.max(1)
                  
                  input_embedded = self.decoder_embedding( input_symbol )
                  
                  sequence_loss += self.criterion( 
                        decoder_output, target_data[:,symbol_index] )  
            return sequence_loss 
      
      def Conv_decode(self, target_data, pooled_activations, unpool_indices):
            prediction =  self.deconv_decoder( pooled_activations, unpool_indices )
            target_data = target_data.view(-1)
            loss = self.criterion( prediction, target_data )
            return loss