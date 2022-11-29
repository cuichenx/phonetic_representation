import torch
from torch import nn


class CLSPooler(nn.Module):
    def __init__(self, num_layers, input_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        # TODO: are we sure we don't need embeddings?
        # TODO: pass the masking if needed
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, tokens, feature_array):
        # prepend [CLS] token to each sequence
        # input: (batch_size, seq_len, 24) - batch_size = 1 for now
        cls_token = torch.ones((1, 1, feature_array.shape[-1]))
        input = torch.cat((cls_token, feature_array), dim=1)

        encoded_input = self.transformer(input)

        # TODO: do we need token embeddings?

        # extract output corresponding to [CLS] token
        pooled_output = encoded_input[:, 0]
        return self.activation(self.linear(pooled_output))
