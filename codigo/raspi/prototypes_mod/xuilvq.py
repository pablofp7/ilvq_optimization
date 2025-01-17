"""
Incremental Learning Vector Quantization
"""

from river import base
from river.utils import dict2numpy
from itertools import chain
from river.utils.math import softmax
from .base_prototypes import BasePrototypes
import time


class XuILVQ(BasePrototypes, base.Classifier):
    """Incremental Learning Vector Quantization classifier (ILVQ).

    This class implements an incremental version of Learning Vector Quantization. ILVQ maintains in memory a set of
    prototypes that try to represent the observed samples, each time ILVQ receives a sample that does not accomplish
    the requirements to be itself a prototype the set of prototypes are updated following the Kohonen rule.
    ILVQ implements a mechanisim to delete potential outdated prototypes based on its age and validity.

    Parameters
    ----------
    alpha_winner
        learning rate for winner prototype
    alpha_runner
        learning rate for runner-up prototype
    age_old
        maximum age of an edge between two prototypes
    gamma
        number of epochs between each prototype cleaning process
    n_prototypes
        number of prototypes to use to claasify a sample

    Examples
    --------
    >>> from river import metrics
    >>> from river import datasets
    >>> from river import evaluate
    >>> from prototypes.xuilvq import XuILVQ
    >>> dataset = datasets.Phishing()
    >>> model = XuILVQ()
    >>> metric = metrics.Accuracy()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 78.62%


    References
    ----------
    [^1]: Xu, Y., Shen, F. and Zhao, J., 2011.
    An incremental learning vector quantization algorithm for pattern classification.
    Neural Computing and Applications,
    21(6), pp.1205-1215.

    """

    def __init__(
            self,
            alpha_winner: float = 0.9,
            alpha_runner: float = 0.1,
            age_old: int = 400,
            gamma: int = 150,
            n_prototypes: int = 5,
            max_pset_size: int = 100,
            target_size: tuple = (80, 90),
            eps: float =  0.000001,
            merge_mode: str = "dbscan", 
            target_percentage: int = 50
    ):
        super().__init__()
        self.alpha_winner = alpha_winner
        self.alpha_runner = alpha_runner
        self.age_old = age_old
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.max_pset_size = max_pset_size
        self.target_size = target_size
        self.eps = eps
        self.merge_mode = merge_mode
        self.target_percentage = target_percentage
        self.clust_time = 0
        self.clust_runs = 0
        

    def learn_one(self, x: dict, y) -> base.Classifier:
        """
        Update the model with a set of features x and a label y.

        Parameters
        ----------
        x
            A dictionary of features
        y
            The class label

        Returns
        -------
        self

        """
        current_epoch = self.epoch #next(self.epoch)
        self.epoch += 1
        x = dict2numpy(x)
        
        time_clust = 0
        
        if len(self.buffer) <= 2:
            self.buffer.append(x, y)

        else:
            try:
                s1, d_s1, _, s2, d_s2, _ = chain(*self.buffer.find_nearest(x, n=2))
            except ValueError:  # temporal, the river check_estimator function needs a None instead a ValueError
                return self
            threshold_s1, threshold_s2 = self.buffer.compute_threshold(s1), self.buffer.compute_threshold(s2)
            if (y not in self.buffer.classes) or (d_s1 > threshold_s1) or (d_s2 > threshold_s2):  # new prototype
                self.buffer.append(x, y)
                return self

            if (s1, s2) not in self.buffer.edges and (s2, s1) not in self.buffer.edges:  # non-directed edge
                self.buffer.create_edge(s1, s2)

            self.buffer.update_edge_age(s1)
            self.buffer.update_winner_m(s1)
            self.buffer.update_positions(x, y, s1=s1, alpha_winner=self.alpha_winner, alpha_runner=self.alpha_runner)
            self.buffer.delete_old_edges(age_old=self.age_old)
            self.alpha_winner = 1/(self.buffer.prototypes[s1]['m'])  # adaptive learning rate winner
            self.alpha_runner = 1/(100 * self.buffer.prototypes[s1]['m'])  # adaptive learning rate runner-up

            start_time = time.perf_counter()
            if len(self.buffer) > self.max_pset_size:
                if "dbscan" in self.merge_mode:
                    self.eps = self.buffer.dbscan_prototypes(max_prototypes=self.max_pset_size, target_range=self.target_size, eps_initial=self.eps)
                elif "kmeans" in self.merge_mode:
                    self.buffer.kmeans_prototypes(max_prototypes=self.max_pset_size, target_percentage=self.target_percentage)
                else: 
                    self.buffer.kmeans_prototypes(max_prototypes=self.max_pset_size, target_percentage=self.target_percentage)

                self.buffer.rebuild_neighborhoods()
                
                self.clust_time += time.perf_counter() - start_time
                self.clust_runs += 1
                
            # if current_epoch % self.gamma == 0:
            #     self.buffer.denoise()
            

        return self

    def predict_one(self, x: dict):
        """
        Predict the label for a dictionary of features x.

        Parameters
        ----------
        x
            A dictionary of features

        Returns
        -------
        y_hat
            The predicted label for the input sample

        """
        if len(self.buffer) == 0 or len(x) != self.buffer.n_features:
            return None  # non fitted predictor

        x = dict2numpy(x)
        nearest = chain(*self.buffer.find_nearest(x, n=min(self.n_prototypes, len(self.buffer))))
        nearest = list(nearest)
        y_hat = {c: 0.0 for c in self.buffer.classes}

        for p in zip(nearest[::3], nearest[1::3], nearest[2::3]):
            s, distance, y = p
            if distance == 0:
                y_hat = {c: 0.0 for c in self.buffer.classes}
                y_hat[y] = 1
                return y_hat
            y_hat[y] += 1/distance

        y_hat_p = softmax(y_hat)
        y_hat = max(y_hat_p, key=y_hat_p.get)
        

        
        
        return y_hat

