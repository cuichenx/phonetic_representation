import panphon2
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances
import multiprocess as mp
from torch.utils.data import DataLoader

class IntrinsicEvaluator:
    def __init__(self):
        self.dist_cache = {}

    def compute_panphon_distance(self, y, data):
        fed = panphon2.FeatureTable().feature_edit_distance
        return [fed(x, y) for x, _ in data]

    def __call__(self, model, data_loader, key):
        data = data_loader.dataset
        single_data_loader = DataLoader(dataset=data, 
                                        batch_size=1, 
                                        collate_fn=lambda x: [y[1] for y in x])
        # compute cosine distances for embeddings
        data_embd = [
            model(x).cpu().detach().numpy()
            for x in single_data_loader
        ]
        data_embd = [x for y in data_embd for x in y]
        data_sims = cosine_distances(data_embd)
        data_sims = np.ravel(data_sims)

        # check if dists for key is already stored, if not compute
        if key is not None and key in self.dist_cache:
            data_dists_true = self.dist_cache[key]
        else:
            # parallelization
            with mp.Pool() as pool:
                data_dists_true = pool.map(
                    lambda y: self.compute_panphon_distance(y[0], data), data)
                # flatten
                data_dists_true = [x for y in data_dists_true for x in y]
        if key is not None and key not in self.dist_cache:
            self.dist_cache[key] = data_dists_true

        parson_corr, _pearson_p = pearsonr(data_sims, data_dists_true)
        spearman_corr, _spearman_p = spearmanr(data_sims, data_dists_true)
        return parson_corr, spearman_corr

        

