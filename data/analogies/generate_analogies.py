import argparse
import random
from collections import defaultdict
from functools import lru_cache

import panphon

ft = panphon.FeatureTable()
FEATURE_NAMES = ft.fts('a').names
NUM_FEATURES = len(FEATURE_NAMES)

'''
Example command: 
python ./data/analogies/generate_analogies.py --lang_codes uz \
--vocab_file data/vocab_uz.txt --output_file data/analogies/uz100_2.txt \
--num_analogies 100 --num_perturbations 2
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_codes', nargs='+')
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--num_analogies', type=int, help='size of the resulting analogy dataset')
    parser.add_argument('--num_perturbations', type=int, help='number phonemes to perturb. '
                                                              'eg. sik : zik <=> ʂik : ʐik is one perturbation,'
                                                              '    sik : zix <=> ʂic : ʐiç is two perturbations.')

    return parser.parse_args()


class PhonemeAnalogy:
    def __init__(self, symbols, tokens, output_file, num_perturbations=1):
        self.single_perturbation_pairs = find_single_perturbation_pairs(symbols)
        self.all_tokens = tokens
        self.output_file = output_file
        self.num_perturbations = num_perturbations


    def find_single_perturbation_pairs(self, symbols):
        feature_names = ft.fts('a').names
        edges = [[] for _ in feature_names]

        feature_vectors = defaultdict(list)
        for x in symbols:
            feature_vectors[tuple(ft.fts(x).numeric())].append(x)
        for feat_i, feat_name in enumerate(feature_names):
            print("=== pairs that only differ in", feat_name, feat_i)
            for vec, symb in feature_vectors.items():
                if vec[feat_i] == -1:
                    partner = list(vec)
                    partner[feat_i] = 1
                    partner = tuple(partner)
                    if partner in feature_vectors:
                        partner_symb = feature_vectors[partner]
                        symb_str, partner_symb_str = '/'.join(symb), '/'.join(partner_symb)
                        print(symb_str, '<=>', partner_symb_str)
                        edges[feat_i].append((symb_str, partner_symb_str))
        return edges

    @lru_cache(None)
    def has_perturbations(self, phoneme):
        return len(self.get_all_perturbations(phoneme)) > 0

    def get_perturbation(self, phoneme, feat_i):
        perturbations = set()
        for ph1, ph2 in self.single_perturbation_pairs[feat_i]:
            if phoneme in ph1.split("/"):
                perturbations.add((ph2, '+'))
            elif phoneme in ph2.split("/"):
                perturbations.add((ph1, '-'))
        return perturbations

    @lru_cache(None)
    def get_all_perturbations(self, phoneme):
        perturbations = []
        for feat_i in range(NUM_FEATURES):
            perturbs = self.get_perturbation(phoneme, feat_i)
            if len(self.single_perturbation_pairs[feat_i]) >= 2:
                # there is at least one other pair of the same type
                for p, pn in perturbs:
                    perturbations.append((p, feat_i, pn))
        return perturbations

    def generate_analogy(self, w1, num_perturbations):
        w2, w3, w4 = w1, w1, w1
        # w1 is a randomly sampled real word
        # sample one phoneme ph1 from w1, and sample two perturbations of the same kind, one of which uses ph1
        # for example, if w1 contains t, the two perturbations might be: t <-> d and f <-> v
        # in the same position of t in w1, w2 gets d, w3 gets f, and w4 gets v


        phoneme_idx_perturbed = set()
        for pi in range(num_perturbations):
            retry_counter = 0
            phoneme_idx = random.choice(range(len(w1)))
            while (phoneme_idx in phoneme_idx_perturbed or not self.has_perturbations(w1[phoneme_idx])):
                retry_counter += 1
                if retry_counter > 100:
                    print("retried 100 times with no success:", w1)
                    return None
                phoneme_idx = random.choice(range(len(w1)))
            phoneme_idx_perturbed.add(phoneme_idx)
            perturbations = self.get_all_perturbations(w1[phoneme_idx])
            # e.g. w1[phoneme_idx] is z
            # perturbations is {('s', 8), ('d', 3), ('ð', 13)}
            w2_char, perturb_type, plus_minus = random.choice(perturbations)
            w3_char, w4_char = w2_char, w2_char
            while w3_char == w2_char or w4_char == w2_char:
                w3_char, w4_char = random.choice(self.single_perturbation_pairs[perturb_type])
                # TODO: will we need to choose w3 and w4 so that they have the same syl feature as w1 and w2?
                # TODO: because if not, we get tuples like ulm	ylm	ŋlm	ɲlm
            if plus_minus == '-':
                w3_char, w4_char = w4_char, w3_char
            w2 = w2[:phoneme_idx] + random.choice(w2_char.split('/')) + w2[phoneme_idx+1:]
            w3 = w3[:phoneme_idx] + random.choice(w3_char.split('/')) + w3[phoneme_idx+1:]
            w4 = w4[:phoneme_idx] + random.choice(w4_char.split('/')) + w4[phoneme_idx+1:]

        return w2, w3, w4

    def run(self, num_analogies):
        for i in range(num_analogies):
            res = None
            while res is None:
                w1 = ''
                while not (3 <= len(w1) <= 8):
                    w1 = random.choice(self.all_tokens)
                res = self.generate_analogy(w1, self.num_perturbations)

            w2, w3, w4 = res
            with open(self.output_file, 'a') as f:
                f.write("\t".join([w1, w2, w3, w4]) + '\n')


def find_single_perturbation_pairs(symbols):
    feature_names = ft.fts('a').names
    edges = [[] for _ in feature_names]

    feature_vectors = defaultdict(list)
    for x in symbols:
        feature_vectors[tuple(ft.fts(x).numeric())].append(x)
    for feat_i, feat_name in enumerate(feature_names):
        print("=== pairs that only differ in", feat_name, feat_i)
        for vec, symb in feature_vectors.items():
            if vec[feat_i] == -1:
                partner = list(vec)
                partner[feat_i] = 1
                partner = tuple(partner)
                if partner in feature_vectors:
                    partner_symb = feature_vectors[partner]
                    symb_str, partner_symb_str = '/'.join(symb), '/'.join(partner_symb)
                    print(symb_str, '<=>', partner_symb_str)
                    edges[feat_i].append((symb_str, partner_symb_str))
    return edges


if __name__ == '__main__':
    args = parse_args()
    with open(args.vocab_file) as f:
        symbols = f.read().split()

    tokens = []
    for lang in args.lang_codes:
        with open(f"data/ipa_tokens_{lang}.txt") as f:
            tokens.extend(f.read().split())

    pa = PhonemeAnalogy(symbols, tokens,
                        args.output_file, num_perturbations=args.num_perturbations)
    pa.run(args.num_analogies)


