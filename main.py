from pathlib import Path

import pandas as pd

from utils import get_logger, load_config
from data import normalize_title_date, sort_filter_clusters, compute_cluster_info, get_top_n_cluster_labels
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

    """
    # Normalize data
    normalize_title_date(c.data.raw, c.resources.omit_strings, c.data.normalized)

    # Vectorize
    df = pd.read_csv(c.data.normalized, lineterminator="\n")
    logger.info(f"Got {df.shape[0]} titles from file: {c.data.normalized}.")
    df = df[df["social_score"] > 0]
    logger.info(f"Got {df.shape[0]} titles after keeping only those with a social score over 0.")
    vectorizer = SentenceTransformerVectorizer(c.vectorizer.model_name_or_path, c.data.cache)
    embeddings = vectorizer.vectorize(df)

    # Cluster
    clusterer = HdbScanClusterer()
    if num_titles:
        clusters = clusterer.cluster(embeddings[:num_titles], df[:num_titles], sort=True, **c.clusterer.toDict())
    else:
        clusters = clusterer.cluster(embeddings, df, sort=True, **c.clusterer.toDict())
    clusters.to_csv(c.data.clustered, index=False)

    # Post-process clusters
    clusters = sort_filter_clusters(clusters)
    clusters.to_csv(c.data.clustered, index=False)
    cluster_scores = compute_cluster_social_scores(clusters)
    cluster_scores.to_csv(c.data.cluster_scores, index=False)
    cluster_labels = cluster_scores["cluster_label"][:top_n_clusters].tolist()
    """

    clusters = pd.read_csv(c.data.clustered, low_memory=False, lineterminator="\n")
    cluster_info = compute_cluster_info(clusters)
    cluster_info.to_csv(c.data.cluster_info, index=False)

    cluster_labels = get_top_n_cluster_labels(cluster_info, top_n_clusters=top_n_clusters)

    # Describe
    #cluster_descriptions = describe(clusters, cluster_labels=cluster_labels)
    #cluster_descriptions.to_csv(c.data.cluster_descriptions, index=False)
    cluster_descriptions = pd.read_csv(c.data.cluster_descriptions)
    cluster_descriptions.sort_values(by=["social_score"], inplace=True, ascending=False)
    cluster_descriptions = cluster_descriptions[:50]

    df_ = cluster_descriptions.merge(right=cluster_info, on=["cluster_label"], how="left")
    df_.drop(columns=["social_score_y"], inplace=True)
    df_.rename(columns={"social_score_x": "social_score"}, inplace=True)
    #df_ = cluster_info.merge(right=cluster_descriptions, how="outer")
    df_.to_csv(c.data.cluster_viz_data, index=False)
