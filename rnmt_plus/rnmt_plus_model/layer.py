from torch import nn
from rnmt_plus_model.relational_rnn import RelationalRNN_module

class EncoderLayer(nn.Module):

    def __init__(self, d_embed, d_hidden):
        super(EncoderLayer, self).__init__()
        self.encoder_relrnn = RelationalRNN_module(d_embed, d_hidden, head_size =int(2500/4),num_heads=4,gate_style='memory',attention_mlp_layers=5,dropout_p=0.3)

    def forward(self, enc_input):
        enc_states, _ = self.encoder_relrnn(enc_input)
        return enc_states

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_embed, d_hidden):
        super(DecoderLayer, self).__init__()
        self.decoder_relrnn = RelationalRNN_module(d_embed, d_hidden, head_size =int(2500/4),num_heads=4,gate_style='memory',attention_mlp_layers=5,dropout_p=0.3)

    def forward(self, dec_input, dec_init_state):
        dec_states, dec_last_hidden = self.decoder_relrnn(dec_input, dec_init_state)
        return dec_states, dec_last_hidden
