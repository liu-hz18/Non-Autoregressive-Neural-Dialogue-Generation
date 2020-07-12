import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'


def tensor_index(atensor, value):
    b = torch.tensor([value], device=atensor.device)
    pos = torch.nonzero(torch.eq(atensor, b), as_tuple=False).squeeze(1)
    p = -1
    try:
        p = pos[0]
    except:
        pass
    return p


class BLEUMetric(object):
    def __init__(self, id2vocab, ignore_smoothing_error=False):
        self.id2vocab = id2vocab
        self.vocab_len = len(id2vocab)
        self.pad_id = id2vocab.index(PAD_TOKEN)
        # self.eos_id = id2vocab.index(EOS_TOKEN)
        self.ignore_smoothing_error = ignore_smoothing_error
        self._reference = []
        self._candidate = []
        self.smooth = SmoothingFunction()

    def _batch_trim(self, batch_ref, batch_can):
        for data in batch_ref:
            self._reference.append([self._convert_to_words(data)])
        for data in batch_can:
            self._candidate.append(self._convert_to_words(data))

    def _trim_before_target(self, lists, target_id):
        lists = lists[:tensor_index(lists, target_id)]
        return lists

    def _drop_pad(self, lists):
        idx = len(lists)
        while idx > 0 and lists[idx - 1] == self.pad_id:
            idx -= 1
        ids = lists[:idx]
        return ids

    def _convert_to_words(self, id_list):
        ids = self._drop_pad(self._trim_before_target(id_list, target_id=self.pad_id))
        words = list(map(lambda word: self.id2vocab[word] if word < self.vocab_len else '<unk>',
                         ids))
        return words

    def _clip_seq(self, id_list, length):
        return list(map(lambda word: self.id2vocab[word] if word < self.vocab_len else '<unk>',
                        id_list[:length] if length > 0 else id_list))

    def _batch_clip(self, batch_ref, batch_can, can_lenths):
        for data in batch_ref:
            self._reference.append([self._convert_to_words(data)])
        for data, length in zip(batch_can, can_lenths):
            self._candidate.append(self._clip_seq(data, length))

    def _calculate(self):
        try:
            corpus_score = corpus_bleu(self._reference, self._candidate, smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            if not self.ignore_smoothing_error:
                raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
                    usually caused when there is only one sample and the sample length is 1.")
            corpus_score = 0.
        return corpus_score

    def forward(self, references, candidates, lengths=None):
        self._reference, self._candidate = [], []
        if lengths is None:
            self._batch_trim(references, candidates)
        else:
            self._batch_clip(references, candidates, lengths)
        return self._calculate()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DistinctNGram(object):
    """docstring for DistinctNGram"""
    def __init__(self, ngram=2):
        super(DistinctNGram, self).__init__()
        self.ngram = ngram
        self.gram_dict = {}

    def _clip_seq(self, id_list, length):
        return list(map(str, id_list[:length] if length > 0 else id_list))

    def _stat_ngram_in_seq(self, tokens):
        tlen = len(tokens)
        for i in range(0, tlen - self.ngram + 1):
            ngram_token = ' '.join(tokens[i:(i + self.ngram)])
            if self.gram_dict.get(ngram_token) is not None:
                self.gram_dict[ngram_token] += 1
            else:
                self.gram_dict[ngram_token] = 1

    def _batch_stat(self, candidates, lengths):
        for seq, length in zip(candidates, lengths):
            self._stat_ngram_in_seq(self._clip_seq(seq, length))

    def forward(self, candidates, lengths):
        self.gram_dict = {}
        self._batch_stat(candidates, lengths)
        return len(self.gram_dict.keys()) / (sum(self.gram_dict.values()) + 1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
