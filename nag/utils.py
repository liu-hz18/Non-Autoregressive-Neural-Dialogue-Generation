import sys
from inspect import signature
from functools import wraps

import torch
from torch.nn.utils.rnn import pad_sequence


class PadCollate(object):
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, pad_id=0, device=None):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.pad_id = pad_id
        self.device = device

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, tensor)
        reutrn:
            xs - a LongTensor of all srcs in batch after padding
            ys - a LongTensor of all tgts in batch after padding
        """
        # find longest sequence
        src_lens = torch.LongTensor([item[0].shape[0] for item in batch]).to(self.device)
        tgt_lens = torch.LongTensor([item[1].shape[0] for item in batch]).to(self.device)
        srcs = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=self.pad_id).to(self.device)
        tgts = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=self.pad_id).to(self.device)
        return srcs, tgts, src_lens, tgt_lens

    def __call__(self, batch):
        return self.pad_collate(batch)


def summary(model, file=sys.stderr):
    def repr(model):
        from functools import reduce
        from torch.nn.modules.module import _addindent
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines
        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)
        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count


def typeassert(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate
