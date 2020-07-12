
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, negative_slope=0.01, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.leaky_relu(self.linear1(x))))
