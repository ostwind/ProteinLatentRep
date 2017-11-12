from CNN_LSTM import *
from encoder import cnn_encoder, rnn_encoder
from decoder import cnn_decoder, AttnDecoderRNN, DecoderRNN

#TODO differences between cho's paper and current implementation, in order of importance
#     parameters and loader currently fitted for protein sequences
#     no segmentation of activation matrix, i.e. no max pooling
#     no highway network
#     gru not bidirectional

class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion):
            super(CharLevel_autoencoder, self).__init__()
            self.encoder_embedding = nn.Embedding(22, 128)
            self.cnn_encoder = cnn_encoder()
            self.rnn_encoder = rnn_encoder()


            self.decoder_hidden_size = 320
            self.decoder_embedding = nn.Embedding(22, self.decoder_hidden_size)
            self.vanilla_decoder = DecoderRNN() 
            self.attention_decoder = AttnDecoderRNN(hidden_size = 320, output_size = 0)
            self.criterion = criterion

            self.seq_len = 81

      def encode(self, data):
            encoder_embedded = self.encoder_embedding(data).unsqueeze(1).transpose(2,3) 
            encoded = self.cnn_encoder.forward(encoder_embedded)
            #print(encoded.data.shape)
            encoded = encoded.squeeze(2)
            #highway
            
            encoder_hidden = self.rnn_encoder.initHidden()
            #print('encoded dimensions', encoded.data.shape, 'encoder_hidden', self.encoder_hidden.data.shape)
            #encoded dimensions torch.Size([64, 25, 27]) encoder_hidden torch.Size([1, 64, 25])
            
            # store encoder outputs as they appear or store seq_len of them on last t?
            #outputs = Variable(torch.zeros(self.seq_len, 25))

            encoder_outputs = Variable(torch.zeros(64, 81, self.decoder_hidden_size))
            for symbol_ind in range(self.seq_len): 
                  output, encoder_hidden = self.rnn_encoder.forward(
                        encoded[:,:,symbol_ind], encoder_hidden)
                  #print(output.data.shape) # (81, 64, 128)
                  encoder_outputs[:, symbol_ind,:] = output[0]
            return encoder_outputs, encoder_hidden

      def decode(self, data, data_onehot, encoder_hidden, encoder_outputs):   
            loss = 0
            decoder_hidden = encoder_hidden
            for amino_acid_index in range(81): 
                  decoder_input = data.data[:, amino_acid_index].unsqueeze(0).transpose(0,1)    
                  #print(decoder_input.data.shape) # 1, 64

                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_embedded = self.decoder_embedding(decoder_input) 
                  decoder_embedded = decoder_embedded.transpose(0,1)
                  #print(decoder_embedded.data.shape)
                  decoder_output, decoder_hidden, attn_weights = self.attention_decoder.forward(
                  decoder_embedded, decoder_hidden, encoder_outputs)
                  
                  #print(decoder_output.data.shape)   # torch.Size([64, 23])
                  target_amino_acid = data_onehot.squeeze(1)
                  target_amino_acid = target_amino_acid[:, :, amino_acid_index].float()

                  #print(decoder_output.data, target_amino_acid) # all predicted tensors = 0 
                  loss += self.criterion(
                        decoder_output.squeeze(0).float(),
                        Variable(target_amino_acid) ) 

            # reconstructed, decoder_hidden = self.vanilla_decoder.forward(
            # decoder_embedded, decoder_hidden)
            return loss #reconstructed, decoder_hidden

# preliminary model
class cnn_autoencoder(nn.Module):
      def __init__(self):
            super(cnn_autoencoder, self).__init__()
            self.encoder = cnn_encoder()
            self.decoder = cnn_decoder()
            # possible symbols, dimension of vector representation for given symbo
            # TODO verify total number of symbols (20 + '_' + ?) 
            self.embedding = nn.Embedding(23, 4)
            
      def encode(self, data):
            char_embeddings = self.embedding(data).unsqueeze(1).transpose(2,3) 
            encoded, unpool_indices = self.encoder.forward(char_embeddings)
            return encoded, unpool_indices

      def decode(self, data, unpool_indices):
            reconstructed = self.decoder.forward(data, unpool_indices)
            return reconstructed