import torch
import panphon2
import multiprocess as mp
import numpy as np
from random import randint, choice
from collections import Counter
from scipy.stats import bernoulli
from torch.utils.data import Dataset

class NearestNeighborDataset(Dataset):
    def __init__(self, data, p=0.1, min_occur=5):
        super().__init__()

        self.data = data
        # self.k = k

        self.ft = panphon2.FeatureTable()
        self.dist = self.ft.feature_edit_distance
        self.length = len(self.data)

        # build vocab
        ipa = set(list(map(lambda x: x[0], self.data)))
        counter = Counter()
        for word in ipa:
            tokens = self.ft.phonemes(word)
            counter.update(tokens)
        
        self.vocab = [w for w, cnt in counter.items() if cnt >= min_occur]
        self.transforms = ['ins', 'del', 'sub']
        self.p = p
    
    def perturb(self, idx):
        word = self.data[idx][0]
        phones = self.ft.phonemes(word)
        perturbed_phones = []
        for i in range(len(phones)):
            if bernoulli.rvs(self.p):
                transform = choice(self.transforms)
                if transform == 'ins':
                    tok = choice(self.vocab)
                    perturbed_phones.append(phones[i])
                    perturbed_phones.append(tok)
                if transform == 'del':
                    perturbed_phones.append('')
                elif transform == 'sub':
                    tok = choice(self.vocab)
                    perturbed_phones.append(tok)
            else:
                perturbed_phones.append(phones[i])
        perturbed_word = "".join(perturbed_phones)

        # sometimes deletion results in an empty string
        if len(perturbed_word) == 0:
            perturbed_word = word

        return (perturbed_word, self.ft.word_to_binary_vectors(perturbed_word))

    def __getitem__(self, idx):
        neg_idx = randint(0, self.length-1)
        if neg_idx == idx:
            while neg_idx == idx:
                neg_idx = randint(0, self.length-1)

        anchor_ipa = self.data[idx][0]
        negative_ipa = self.data[neg_idx][0]

        anchor_fv = self.data[idx][1]
        negative_fv = self.data[neg_idx][1]

        # positive example
        positive_ipa, positive_fv = self.perturb(idx)

        return {
            'anchor': (anchor_ipa, anchor_fv),
            'positive': (positive_ipa, positive_fv),
            'negative': (negative_ipa, negative_fv)
        }


    def __len__(self):
        return len(self.data)