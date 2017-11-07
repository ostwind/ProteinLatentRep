from CNN_LSTM import *

from cnn_encoder import cnn_encoder
from cnn_decoder import cnn_decoder


class cnn_autoencoder(nn.Module):
      def __init__(self):
            super(cnn_autoencoder, self).__init__()
            self.encoder = cnn_encoder()
            self.decoder = cnn_decoder()
            # possible symbols, dimension of vector representation for given symbo
            # TODO verify total number of symbols (20 + '_' + ?) 
            self.embedding = nn.Embedding(23, 4)

      def encode(self, data):
            char_embeddings = self.embedding(data).unsqueeze(1) 
            char_embeddings = char_embeddings.transpose(2,3)
            encoded, unpool_indices = self.encoder.forward(char_embeddings)
            return encoded, unpool_indices

      def decode(self, data, unpool_indices):
            reconstructed = self.decoder.forward(data, unpool_indices)
            return reconstructed