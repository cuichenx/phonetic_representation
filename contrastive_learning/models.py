import torch 
from torch import nn 
import panphon2


class LSTM_Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, device, feature_size=24, bidirectional=True):
        super().__init__()

        self.encoder = torch.nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # TODO: try linear and non-linear projection head
        # after training is completed the projection head is discarded (https://arxiv.org/pdf/2002.05709.pdf)
        if bidirectional:
            self.out_dim = 2 * hidden_size
        else:
            self.out_dim = hidden_size
            
        self.proj_head = torch.nn.Sequential(torch.nn.Dropout(0.1),
                                             torch.nn.Linear(self.out_dim, self.out_dim),
                                             torch.nn.BatchNorm1d(self.out_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(self.out_dim, 64, bias=False))

        self.use_proj_head = True
        
        self.to(device)
        self.device = device

    def proj_head_on(self):
        self.use_proj_head = True

    def proj_head_off(self):
        self.use_proj_head = False

    def forward(self, x):
        x_pad = torch.nn.utils.rnn.pad_sequence(
                    [torch.Tensor(x_0) for x_0 in x],
                    batch_first=True, padding_value=-1.0,
                ).to(self.device)

        output, (_, _) = self.encoder(x_pad)
        output = output[:, -1, :]

        if self.use_proj_head:
            output = self.proj_head(output)
        
        return output