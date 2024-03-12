import torch
import torch.nn as nn
import torch.nn.functional as F


import random


class Attention(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()

        self.hid_dim = hid_dim
        self.activation = nn.Tanh()

    def forward(self, hidden, encoder_output):
        
        # [B * L * H] -> [B * H * L]
        encoder_output_permuted = encoder_output.permute(0, 2, 1)

        # [N * B * H] -> [B * N * H]
        hidden_permuted = hidden.permute(1, 0, 2)

        # [B * N * H] x [B * H * L] = [B * N * L]
        attention_score = hidden_permuted.bmm(encoder_output_permuted)

        # [B * N * L]
        attention_weights = F.softmax(attention_score, dim=2)

        # [B * N * L] x [B * L * H] = [B * N * H] -> [N * B * H]
        attention_vector = torch.bmm(attention_weights, encoder_output).permute(1, 0, 2)

        hidden_updated = self.activation(attention_vector)

        return hidden_updated

        
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, src):
        
        # src = [batch size, src sent len]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)
        
        embedded = self.dropout(embedded)
        
        output, (hidden, cell) = self.rnn(embedded)
        # embedded = [batch size, src sent len, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        
        # outputs = [batch size, src sent len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # outputs are always from the top hidden layer
        
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
        
        self.dropout = nn.Dropout(p=dropout)

        self.attention = Attention(hid_dim)

        self.activation = nn.ReLU()

    
    def forward(self, input, hidden, cell, encoder_output):
            
            
        # input = [batch size]
        # hidden = [n layers * n directions, batch size,  hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
            
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
            
        input = input.unsqueeze(1)
            
        # input = [batch size, 1]
            
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(input))
            
        # embedded = [batch size, 1, emb dim]
            
        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
            
            
        # output = [batch size, sent len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
            
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [batch size, 1, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        hidden_updated = self.attention(hidden, encoder_output)
            
        output, (hidden, cell) = self.rnn(embedded, (hidden_updated.contiguous(), cell.contiguous()))

        # prediction = [batch size, output dim]
        prediction = self.out(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, attention):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.attentionn = attention
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output, hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_output)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[:, t] if teacher_force else top1)
        
        return outputs
