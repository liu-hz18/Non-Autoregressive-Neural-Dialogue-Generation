import os
from tqdm import tqdm
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'


class VocabBulider(object):
    def __init__(self, src_dir, src_files: list, vocab_file=None, min_count=5,
                 update=False, ignore_unk_error=False):
        super(VocabBulider, self).__init__()
        self.src_files = src_files
        self.src_dir = src_dir
        self.min_count = min_count
        self.vocab2id = {}
        self.vocab_file = vocab_file
        self.save_path = os.path.join(self.src_dir, self.vocab_file)
        if os.path.exists(self.save_path) and os.path.getsize(self.save_path) > 0 and not update:
            self.id2vocab = []
            self._read_vocab(self.vocab_file)
        else:
            self.id2vocab = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
            self._build_vocab()
        self.start_id = len(self.id2vocab)
        self.padid = self.id2vocab.index(PAD_TOKEN)

    def _build_vocab(self):
        words_list = []
        for file_name in self.src_files:
            words_list.extend(self._read_file(file_name))
        print('Buliding vocabulary...')
        fdist = FreqDist(words_list)
        sorted_fdist = sorted(fdist.items(), key=lambda x: (x[1], x[0]), reverse=True)
        print(sorted_fdist[:100])
        self.id2vocab.extend([key for key, value in sorted_fdist if value >= self.min_count])
        self._save_vocab(self.vocab_file)
        self.vocab2id = dict(zip(self.id2vocab, range(len(self.id2vocab))))

    def _read_file(self, file_name):
        print(f'Reading src files: {file_name}...')
        with open(os.path.join(self.src_dir, file_name), 'r', encoding='utf-8') as f:
            words_list = word_tokenize(f.read())
        return words_list

    def _save_vocab(self, vocab_file):
        with open(self.save_path, 'w', encoding='utf-8') as f:
            for word in tqdm(self.id2vocab, desc='Saving vocab'):
                f.write(word + '\n')

    def _read_vocab(self, vocab_file):
        print('Reading vocabulary...')
        with open(self.save_path, 'r', encoding='utf-8') as f:
            for word in tqdm(f.readlines(), desc='Reading vocab'):
                self.id2vocab.append(word.rstrip('\n'))
        self.vocab2id = dict(zip(self.id2vocab, range(len(self.id2vocab))))

    def most_common(self, rank):
        return self.id2vocab[:rank]

    def __len__(self):
        return len(self.id2vocab)

    def __getitem__(self, word):
        idx = self.id2vocab.index(UNK_TOKEN)
        try:
            idx = self.vocab2id[word]
        except KeyError as _:
            if self.ignore_unk_error == False:
                raise KeyError(f'word {word} not in word_list!')
        return idx

    def id_to_word(self, idx):
        return self.id2word[idx]
