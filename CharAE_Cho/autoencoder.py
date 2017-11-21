from CharAE_Cho import *
from encoder import cnn_encoder, rnn_encoder
from decoder import cnn_decoder, AttnDecoderRNN, DecoderRNN

#TODO differences between cho's paper and current implementation, in order of importance
#     no highway network
#     gru not bidirectional

class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion):
            super(CharLevel_autoencoder, self).__init__()
            self.char_embedding_dim = 64
            self.filter_widths = list(range(1, 6)) 
            self.num_filters_per_width = 7
            self.seq_len = 84

            self.encoder_embedding = nn.Embedding(22, self.char_embedding_dim)
            self.cnn_encoder = cnn_encoder(
            filter_widths = self.filter_widths,
            num_filters_per_width = self.num_filters_per_width,
            char_embedding_dim = self.char_embedding_dim,
            seq_len = self.seq_len)
            
            self.decoder_hidden_size = len(self.filter_widths) * self.num_filters_per_width
            self.rnn_encoder = rnn_encoder( 
            hidden_size = self.decoder_hidden_size )

            self.rnn_emits_len = 21
            self.decoder_embedding = nn.Embedding(22, self.decoder_hidden_size)
            self.attention_decoder = AttnDecoderRNN(
                  hidden_size = self.decoder_hidden_size, output_size = self.rnn_emits_len)
            self.criterion = criterion

      def encode(self, data, collect_filters = False):
            encoder_embedded = self.encoder_embedding(data).unsqueeze(1).transpose(2,3) 
            encoded = self.cnn_encoder.forward(encoder_embedded, collect_filters)
            encoded = encoded.squeeze(2)
            #highway
            
            encoder_hidden = self.rnn_encoder.initHidden()
            #print('encoded dimensions', encoded.data.shape, 'encoder_hidden', self.encoder_hidden.data.shape)
            #encoded dimensions torch.Size([64, 25, 27]) encoder_hidden torch.Size([1, 64, 25])
            
            # store encoder outputs as they appear or store seq_len of them on last t?
            #outputs = Variable(torch.zeros(self.seq_len, 25))

            encoder_outputs = Variable(torch.zeros(64, self.rnn_emits_len, self.decoder_hidden_size))
            for symbol_ind in range(self.rnn_emits_len): 
                  output, encoder_hidden = self.rnn_encoder.forward(
                        encoded[:,:,symbol_ind], encoder_hidden)
                  #print(output.data.shape) # (81, 64, 128)
                  encoder_outputs[:, symbol_ind,:] = output[0]
            return encoder_outputs, encoder_hidden

      def decode(self, data, data_onehot, encoder_hidden, encoder_outputs):   
            loss = 0
            decoder_hidden = encoder_hidden
            for amino_acid_index in range(self.seq_len): 
                  decoder_input = data.data[:, amino_acid_index].unsqueeze(0).transpose(0,1)    
                  #print(decoder_input.data.shape) # 1, 64

                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_embedded = self.decoder_embedding(decoder_input) 
                  decoder_embedded = decoder_embedded.transpose(0,1)
                  decoder_output, decoder_hidden, attn_weights = self.attention_decoder.forward(
                  decoder_embedded, decoder_hidden, encoder_outputs)
                  
                  #print(decoder_output.data.shape)   # torch.Size([64, 23])
                  target_amino_acid = data_onehot.squeeze(1)
                  target_amino_acid = target_amino_acid[:, :, amino_acid_index].float()

                  #print(decoder_output.data, target_amino_acid) # all predicted tensors = 0 
                  loss += self.criterion(
                        decoder_output.squeeze(0).float(),
                        Variable(target_amino_acid) ) 
            return loss 

# preliminary model
class cnn_autoencoder(nn.Module):
      def __init__(self):
            super(cnn_autoencoder, self).__init__()
            self.encoder = cnn_encoder()
            self.decoder = cnn_decoder()
            # possible symbols, dimension of vector representation for given symbo
            # TODO verify total number of symbols (20 + '_' + ?) 
            self.embedding = nn.Embedding(22, 4)
            
      def encode(self, data):
            char_embeddings = self.embedding(data).unsqueeze(1).transpose(2,3) 
            encoded, unpool_indices = self.encoder.forward(char_embeddings)
            return encoded, unpool_indices

      def decode(self, data, unpool_indices):
            reconstructed = self.decoder.forward(data, unpool_indices)
            return reconstructed