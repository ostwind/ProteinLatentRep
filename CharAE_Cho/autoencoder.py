from CharAE_Cho import *
from encoder import cnn_encoder, rnn_encoder
from decoder import rnn_decoder, cnn_decoder
import random 

class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion, seq_len, layers ):
            super(CharLevel_autoencoder, self).__init__()
            self.batch_size = 64
            self.layers = layers
            self.seq_len = seq_len
            self.emit_len = seq_len#//3 #CR|R C|C conv encoder -> pooled activations 
            if layers == 'R_R': 
                  self.emit_len = seq_len

            self.layers = layers    
            self.char_embedding_dim = 20 
            self.encoder_embedding = nn.Embedding(23, self.char_embedding_dim)
            
            self.filter_widths = list(range(1, 6))
            self.filter_config = [20, 20, 20, 20, 10]  #for CR_R and C_C
            #self.filter_config = [10, 10, 20, 20] 

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

      def GRU_encode(self, activations, inference):
            if self.layers == 'R_R':
                  activations = activations.transpose(1,3).contiguous().view(
                        self.batch_size, self.emit_len, -1)
            
            else: 
                  activations = activations.transpose(1, 2).contiguous()

            encoder_hidden = self.rnn_encoder.initHidden( self.batch_size )
            encoder_outputs = Variable(torch.zeros(self.batch_size, self.emit_len, 2*self.decoder_hidden_size))
            
            output, encoder_hidden = self.rnn_encoder.forward(
                       activations, encoder_hidden)

            # k = 50
            # output = output.contiguous().view(self.batch_size, -1)
            # # hidden state sparse
            # global indices  
            # large_vals, indices = output.sort(1, descending = True)
            # # zero all values not in the top k 
            # output[ indices > k ] = 0 
            # # set gradients of zeroed neurons to zero 
            # if not inference:
            #       h = output.register_hook( hook )
            #output = output.view(self.batch_size, 28, 160)

            return output, encoder_hidden 

      def NMT_encode(self, embedded, inference = False ):
            activations, _ = self.Conv_encode(embedded)
            encoder_outputs, encoder_hidden = self.GRU_encode(activations, inference)
            return encoder_outputs, encoder_hidden

      def GRU_decode(self, target_data, encoder_hidden, encoder_outputs,
       dont_return = True, attention = True):  
            decoder_hidden = encoder_hidden
            input_embedded = Variable(torch.LongTensor([17]).repeat(self.batch_size)) # SOS token
            input_embedded = self.decoder_embedding( input_embedded )
            
            sequence_loss = 0
            decoder_outputs = Variable( torch.FloatTensor(self.batch_size, self.seq_len, 23 ) )
            for symbol_index in range(self.seq_len): 
                  
                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_output, decoder_hidden, attn_weights = self.attention_decoder.forward(
                  input_embedded, decoder_hidden, encoder_outputs, attention, self.batch_size)
                  
                  values, symbol_pred = decoder_output.max(1)
                  
                  input_embedded = self.decoder_embedding( symbol_pred )
                  
                  sequence_loss += self.criterion( 
                       decoder_output, target_data[:,symbol_index] )  
                  decoder_outputs[:, symbol_index, :] = decoder_output

            return sequence_loss 
      
      def Conv_decode(self, target_data, pooled_activations, unpool_indices):
            prediction =  self.deconv_decoder( pooled_activations, unpool_indices )
            target_data = target_data.view(-1)
            loss = self.criterion( prediction, target_data )
            return loss

def hook(grad):
      grad_clone = grad.clone()
      grad_clone[ indices > 50 ] = 0
      return grad_clone


# class L1Penalty(Function):

#     @staticmethod
#     def forward(ctx, input, l1weight):
#         ctx.save_for_backward(input)
#         ctx.l1weight = l1weight
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_variables
#         grad_input = input.clone().sign().mul(self.l1weight)
#         grad_input += grad_output
#         return grad_input