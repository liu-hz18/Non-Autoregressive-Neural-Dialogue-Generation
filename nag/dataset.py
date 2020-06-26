import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

SOS_ID = 2
EOS_ID = 3


class OpenSubDataset(Dataset):

    def __init__(self, data_dir, vocab_bulider):
        '''
        :data_dir:   string, data dir
        :data_files: List, [filename1, filename2, ...]
        '''
        super(OpenSubDataset, self).__init__()
        self.data_dir = data_dir
        self.file_name_prefix = 'opensub_pair_'
        self.file_classes = ['train', 'dev', 'test']
        # self.file_classes = ['dev', 'test']
        self.vocab_bulider = vocab_bulider
        self.vocab_bulider.ignore_unk_error = True
        self.posts = []
        self.reps = []
        self._prepare_dataset()
        assert len(self.posts) == len(self.reps), 'length of posts DON\'T MATCH length of reps'

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        return self.posts[idx], self.reps[idx]

    def _read_data_file(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        sentences = []
        show_name = file_name.split('.')[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc=f'reading: {show_name}'):
                sen = torch.LongTensor([SOS_ID] + list(map(lambda x: self.vocab_bulider[x], \
                                                       word_tokenize(line.strip()))) + [EOS_ID])
                sentences.append(sen)
        return sentences

    def _prepare_dataset(self):
        for file_class in self.file_classes:
            file_prefix = self.file_name_prefix + file_class + '.'
            self._read_post_file(file_prefix)
            self._read_reps_file(file_prefix)

    def _read_post_file(self, file_prefix):
        self.posts.extend(self._read_data_file(file_prefix + 'post'))

    def _read_reps_file(self, file_prefix):
        self.reps.extend(self._read_data_file(file_prefix + 'response'))


class IMSDBDataset(Dataset):

    def __init__(self, data_dir, vocab_bulider, max_seq_length=20):
        '''
        :data_dir:   string, data dir
        '''
        super(IMSDBDataset, self).__init__()
        self.data_dir = data_dir
        self.vocab_bulider = vocab_bulider
        self.vocab_bulider.ignore_unk_error = True
        self.file_name_prefix = 'imsdb_'
        self.file_classes = ['lower']
        self.posts = []
        self.reps = []
        self.max_seq_length = max_seq_length
        self._prepare_dataset()
        assert len(self.posts) == len(self.reps), 'length of posts DON\'T MATCH length of reps'

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        return self.posts[idx], self.reps[idx]

    def _read_data_file(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc=f'reading: {file_name}'):
                sen = torch.LongTensor([SOS_ID] + list(map(lambda x: self.vocab_bulider[x], \
                                                       line.strip().split())) + [EOS_ID])
                sentences.append(sen)
        return sentences

    def _prepare_dataset(self):
        for file_class in self.file_classes:
            file_prefix = self.file_name_prefix + file_class + '.'
            self.posts.extend(self._read_data_file(file_prefix + 'post'))
            self.reps.extend(self._read_data_file(file_prefix + 'response'))
