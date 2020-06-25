import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
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
        self.eos_id = id2vocab.index(EOS_TOKEN)
        self.ignore_smoothing_error = ignore_smoothing_error
        self._reference = []
        self._candidate = []
        self.smooth = SmoothingFunction()

    def _batch_trim(self, batch_ref, batch_can):
        for data in batch_ref:
            self._reference.append(self._convert_to_words(data))
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
        words = list(map(lambda word: self.id2vocab[word], ids))
        return words

    def _calculate(self):
        scores = []
        for ref, can in zip(self._reference, self._candidate):
            try:
                corpus_score = corpus_bleu([[ref]], [can], smoothing_function=SmoothingFunction().method3)
            except ZeroDivisionError as _:
                if not self.ignore_smoothing_error:
                    raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
                        usually caused when there is only one sample and the sample length is 1.")
                corpus_score = 0
            scores.append(corpus_score)
        return np.mean(scores)

    def forward(self, references, candidates):
        self._reference, self._candidate = [], []
        self._batch_trim(references, candidates)
        return self._calculate()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
