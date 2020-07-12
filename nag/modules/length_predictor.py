import torch
from torch import nn
from torch.nn import functional as F


class LengthPredictor(nn.Module):
    def __init__(self, embed_size, min_value=-20, max_value=20, max_length=250,
                 tau=1, diffrentable=False, device=None):
        super(LengthPredictor, self).__init__()
        self.embed_size = embed_size
        self.out_size = max_value - min_value + 1
        self.pooling_layer = nn.AdaptiveMaxPool1d(output_size=1)
        self.mlp = nn.Linear(self.embed_size, self.out_size)
        self.min_value = min_value
        self.device = device
        self.max_length = max_length
        self.diffrentable = diffrentable
        self.min_length = torch.Tensor([1]).to(self.device)
        if self.diffrentable:
            max_length = max_length + max_value + 1
            range_vec_i = torch.arange(max_length).float().to(self.device)
            range_vec_j = torch.arange(max_length).float().to(self.device)
            distance_mat = F.softmax(-torch.abs(range_vec_i[None, :] - range_vec_j[:, None]) / tau, dim=-1)
            self.W = nn.Parameter(distance_mat, requires_grad=False)

    def forward(self, x, src_length, tgt_length=None):
        '''
        in: B x length x embed_size
        out: B x new_length
        '''
        input_len = x.shape[1]
        out = self.pooling_layer(x.permute(0, 2, 1)).squeeze(2)  # out: B x embed_size
        len_out_prob = self.mlp(out)  # out: B x [-m, m]
        if tgt_length is not None:
            output_len = torch.max(tgt_length)
        else:
            ms = torch.argmax(len_out_prob.detach(), dim=1) + src_length + self.min_value   # out: B
            output_len = torch.max(ms)
            if output_len < 2:
                output_len = self.min_length
        if self.diffrentable:
            output = x.permute(0, 2, 1)\
                      .matmul(self.W[:input_len, :output_len])\
                      .permute(0, 2, 1)
        else:
            output = self._soft_copy(
                x.cpu().detach().numpy(), input_len, output_len.item())
        return output, len_out_prob

    def _soft_copy(self, hidden, input_len, output_len):
        output = torch.Tensor(
            [self._soft_copy_per_sentence(seq, input_len, output_len) for seq in hidden])\
            .to(self.device)
        return output

    def _soft_copy_per_sentence(self, hidden, input_len, output_len):
        return [hidden[i*input_len // output_len] for i in range(output_len)]
