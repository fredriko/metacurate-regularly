from time import time

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_logger

logger = get_logger("cluster")


class HdbScanClusterer(object):

    def __init__(self):
        pass

    def __str__(self) -> str:
        return "HdbScanClusterer"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
        distance_matrix = np.float64(1 - cosine_similarity(embeddings))
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix

    @staticmethod
    def cluster(array_like: np.ndarray, data: pd.DataFrame, data_column: str = "title_normalized", sort: bool = False,
                **kwargs) -> pd.DataFrame:
        """
        :param array_like: Either vectors each of which represents a data point, or a distance matrix with pairwise
        distances for the vectors. In the latter case, make sure to set "metric" to "precomputed".
        :param data: The dataframe containing the data to be clustered.
        :param data_column: The name of the column that holds the data corresponding to the contents of array_like,
        i.e., the data to be clustered.
        :param sort: Set to True if the resulting dataframe should be sorted according to label and cluster probability.
        :return: A dataframe with columns data_column, "cluster_label", "cluster_probability".

        Useful keyword arguments:

        * cluster_selection_method: "leaf" or "eom".
        * min_cluster_size:
        """
        logger.info(f"Got kwargs: {kwargs}")
        metric = kwargs.get("metric")
        if metric and metric == "precomputed":
            array_like = HdbScanClusterer.compute_distance_matrix(array_like)
        clusterer = hdbscan.HDBSCAN(**kwargs)
        start_time = time()
        clusterer.fit_predict(array_like)
        logger.info(f"Clustered {array_like.shape} data in {round(time() - start_time, 2)} seconds")
        data.insert(loc=0, column="cluster_label", value=clusterer.labels_)
        data.insert(loc=1, column="cluster_probability", value=clusterer.probabilities_)
        if sort:
            data = data.sort_values(["cluster_label", "cluster_probability"], ascending=(False, False))
        return data
