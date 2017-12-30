from CharAE_Cho import *
from encoder import cnn_encoder, rnn_encoder
from decoder import AttnDecoderRNN
import random 

class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion):#, seq_len):
            super(CharLevel_autoencoder, self).__init__()
            self.char_embedding_dim = 64
            self.filter_widths = list(range(1, 8)) 
            self.num_filters_per_width = [40, 40, 80, 80, 40, 40, 40] # too wide leads to memorization
            # TODO experiment with more conv layers
            self.encoder_embedding = nn.Embedding(23, self.char_embedding_dim)
            self.cnn_encoder = cnn_encoder(
            filter_widths = self.filter_widths,
            num_filters_per_width = self.num_filters_per_width,
            char_embedding_dim = self.char_embedding_dim)
            #seq_len = self.seq_len)
            
            self.decoder_hidden_size = int(np.sum(np.array(self.num_filters_per_width)) )#len(self.filter_widths) * self.num_filters_per_width
            self.rnn_encoder = rnn_encoder( 
            hidden_size = self.decoder_hidden_size )

            self.decoder_embedding = nn.Embedding(23, self.decoder_hidden_size)
            self.attention_decoder = AttnDecoderRNN(
                  hidden_size = self.decoder_hidden_size, seq_len = 78//3)
            self.criterion = criterion

      def encode(self, data, seq_len, collect_filters = False):
            encoder_embedded = self.encoder_embedding(data)
            #print(encoder_embedded.data.shape)
            encoder_embedded = encoder_embedded.unsqueeze(1).transpose(2,3) 
            #print(encoder_embedded.data.shape)
            
            encoded = self.cnn_encoder.forward(encoder_embedded, seq_len, collect_filters)
            encoded = encoded.squeeze(2)
      
            encoder_hidden = self.rnn_encoder.initHidden()
            #print('encoded dimensions', encoded.data.shape, 'encoder_hidden', self.encoder_hidden.data.shape)
            #encoded dimensions torch.Size([64, 25, 27]) encoder_hidden torch.Size([1, 64, 25])
            
            # 2 times hidden size for bi-directional gru 
            encoder_outputs = Variable(torch.zeros(64, seq_len//3, 2*self.decoder_hidden_size))
            for symbol_ind in range(seq_len//3): 
                  output, encoder_hidden = self.rnn_encoder.forward(
                        encoded[:,:,symbol_ind], encoder_hidden)
                  #print(output.data.shape) # (81, 64, 128)
                  encoder_outputs[:, symbol_ind,:] = output[0]
            return encoder_outputs, encoder_hidden


      def decode(self, target_data, encoder_hidden, encoder_outputs, seq_len, i = False):   
            use_teacher_forcing = True if random.random() < 0.6 else False
            if type(i) != bool: # given batch  index, then eval mode, no teacher forcing
                  use_teacher_forcing = False
            
            decoder_hidden = encoder_hidden
            input_embedded = Variable(torch.LongTensor([17]).repeat(64))
            # if self.use_cuda:
            #       input_embedded = input_embedded.cuda()
            input_embedded = self.decoder_embedding( input_embedded )
            
            output = []
            for symbol_index in range(seq_len): 
                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_output, decoder_hidden, attn_weights = self.attention_decoder.forward(
                  input_embedded, decoder_hidden, encoder_outputs, seq_len//3)
                  
                  if use_teacher_forcing:
                        input_symbol = Variable(target_data[:, symbol_index])

                  else:
                        values, input_symbol = decoder_output.max(1)
                  input_embedded = self.decoder_embedding( input_symbol )
                  
                  output.append( decoder_output )
            
            predicted = torch.stack(output, dim=2).view(-1, 23)
            loss = self.criterion(
                  predicted,
                  Variable(target_data.view(-1)) )
            return loss 
