from pathlib import Path

import pandas as pd

from utils import get_logger, load_config
from data import normalize_title_date, sort_filter_clusters, compute_cluster_social_scores
from vectorize import SentenceTransformerVectorizer
from dotmap import DotMap
from cluster import HdbScanClusterer
from describe import describe

logger = get_logger("main")

if __name__ == "__main__":
    num_titles = None
    top_n_clusters = 100
    config_file = Path("config.json")
    c: DotMap = load_config(config_file)
    #normalize_title_date(c.data.raw, c.resources.omit_strings, c.data.normalized)

    df = pd.read_csv(c.data.normalized, lineterminator="\n", nrows=None)
    logger.info(f"Got {df.shape[0]} titles from file: {c.data.normalized}.")
    df = df[df["social_score"] > 0]
    logger.info(f"Got {df.shape[0]} titles after keeping only those with a social score over 0.")
    vectorizer = SentenceTransformerVectorizer(c.vectorizer.model_name_or_path, c.data.cache)
    embeddings = vectorizer.vectorize(df)

    clusterer = HdbScanClusterer()
    if num_titles:
        clusters = clusterer.cluster(embeddings[:num_titles], df[:num_titles], sort=True, **c.clusterer.toDict())
    else:
        clusters = clusterer.cluster(embeddings, df, sort=True, **c.clusterer.toDict())
    clusters.to_csv(c.data.clustered, index=False)

    clusters = sort_filter_clusters(clusters)
    clusters.to_csv(c.data.clustered, index=False)
    cluster_scores = compute_cluster_social_scores(clusters)
    cluster_scores.to_csv(c.data.cluster_scores, index=False)
    cluster_labels = cluster_scores["cluster_label"][:top_n_clusters].tolist()

    cluster_descriptions = describe(clusters, cluster_labels=cluster_labels)
    cluster_descriptions.to_csv(c.data.cluster_descriptions, index=False)

    df_ = cluster_scores.merge(right=cluster_descriptions, on="cluster_label", how="outer")
    df_.to_csv("tt.csv", index=False)
