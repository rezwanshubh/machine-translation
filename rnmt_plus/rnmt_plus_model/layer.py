from torch import nn


class EncoderLayer(nn.Module):

    def __init__(self, d_embed, d_hidden):
        super(EncoderLayer, self).__init__()
        self.encoder_relu = nn.ReLU()

    def forward(self, enc_input):
        enc_states, _ = self.encoder_relu(enc_input)
        return enc_states


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_embed, d_hidden):
        super(DecoderLayer, self).__init__()
        self.decoder_relu = nn.ReLU()

    def forward(self, dec_input, dec_init_state):
        dec_states, dec_last_hidden = self.decoder_relu(dec_input, dec_init_state)
        return dec_states, dec_last_hidden
