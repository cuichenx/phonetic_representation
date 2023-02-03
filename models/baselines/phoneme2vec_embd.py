import pickle
import _pickle as cPickle
import gzip

# 40 phoneme embeddings (dimension 50) from Fang et al 2020
# trained using the s2s method from the paper

with gzip.open('models/baselines/phoneme2vec', 'rb') as f:
    phonemes, embeddings = cPickle.load(f)
