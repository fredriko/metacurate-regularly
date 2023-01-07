from dotmap import DotMap

from cluster import cluster
from data import normalize_title_date, sort_filter_clusters, compute_cluster_info, prep_output_directory
from describe import describe
from utils import get_logger, load_config, get_df, get_top_n_cluster_labels
from vectorize import SentenceTransformerVectorizer
from visualize import visualize
from typing import Optional

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = get_logger("main")


def main(config_file: str = "config.json", describe_top_n_clusters: int = 100, visualize_top_n_clusters: int = 50,
         cluster_probability: float = 0.7, height: Optional[int] = None, width: Optional[int] = None) -> None:
    c: DotMap = load_config(config_file)
    """
    # Prep directories
    prep_output_directory(c)

    # Normalize data
    df = normalize_title_date(c.data.raw, c.resources.omit_strings)
    df.to_csv(c.data.normalized, index=False)

    # Vectorize
    logger.info(f"Got {df.shape[0]} titles from file: {c.data.normalized}.")
    df = df[df["social_score"] > 0]
    logger.info(f"Got {df.shape[0]} titles after keeping only those with a social score over 0.")
    vectorizer = SentenceTransformerVectorizer(c.vectorizer.model_name_or_path, c.data.cache)
    embeddings = vectorizer.vectorize(df)

    # Cluster
    clusters = cluster(embeddings, df, sort=True, **c.clusterer.toDict())
    clusters.to_csv(c.data.clustered, index=False)

    # Post-process clusters
    clusters = sort_filter_clusters(clusters, cluster_probability=cluster_probability)
    clusters.to_csv(c.data.clustered, index=False)
    cluster_info = compute_cluster_info(clusters)
    cluster_info.to_csv(c.data.cluster_info, index=False)

    # Describe
    cluster_labels = get_top_n_cluster_labels(cluster_info, top_n_clusters=describe_top_n_clusters)
    cluster_descriptions = describe(clusters, cluster_labels=cluster_labels, cluster_probability=cluster_probability)
    cluster_descriptions.to_csv(c.data.cluster_descriptions, index=False)
    """

    # Create visualization data

    # TODO remove
    cluster_descriptions = get_df(c.data.cluster_descriptions)
    cluster_info = get_df(c.data.cluster_info)

    viz_data = cluster_descriptions.merge(right=cluster_info, on=["cluster_label"], how="left")
    viz_data.sort_values(by=["total_social_score", "start_date", "social_score"], ascending=(False, True, False),
                         inplace=True)
    viz_data.to_csv(c.data.cluster_viz_data, index=False)

    visualize(viz_data, visualize_top_n_clusters, save_html=True, html_file_name=c.data.cluster_viz_html, publish=False,
              height=height)


if __name__ == "__main__":
    config_file = "config.json"
    describe_n_clusters = 300
    visualize_n_clusters = 50
    cluster_probability = 0.8
    height = None
    main(config_file, describe_top_n_clusters=describe_n_clusters, visualize_top_n_clusters=visualize_n_clusters,
         cluster_probability=cluster_probability, height=height)
