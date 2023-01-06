from dotmap import DotMap

from cluster import cluster
from data import normalize_title_date, sort_filter_clusters, compute_cluster_info, get_top_n_cluster_labels
from describe import describe
from utils import get_logger, load_config, get_df
from vectorize import SentenceTransformerVectorizer
from visualize import visualize

logger = get_logger("main")


def main(config_file: str = "config.json", top_n_clusters: int = 100,
         cluster_probability_threshold: float = 0.7) -> None:

    c: DotMap = load_config(config_file)

    # Normalize data
    normalize_title_date(c.data.raw, c.resources.omit_strings, c.data.normalized)

    # Vectorize
    df = get_df(c.data.normalized)
    logger.info(f"Got {df.shape[0]} titles from file: {c.data.normalized}.")
    df = df[df["social_score"] > 0]
    logger.info(f"Got {df.shape[0]} titles after keeping only those with a social score over 0.")
    vectorizer = SentenceTransformerVectorizer(c.vectorizer.model_name_or_path, c.data.cache)
    embeddings = vectorizer.vectorize(df)

    # Cluster
    clusters = cluster(embeddings, df, sort=True, **c.clusterer.toDict())
    clusters.to_csv(c.data.clustered, index=False)

    # Post-process clusters
    clusters = sort_filter_clusters(clusters, cluster_probability_threshold=cluster_probability_threshold)
    clusters.to_csv(c.data.clustered, index=False)
    cluster_info = compute_cluster_info(clusters)
    cluster_info.to_csv(c.data.cluster_info, index=False)

    # Describe
    cluster_labels = get_top_n_cluster_labels(cluster_info, top_n_clusters=top_n_clusters)
    cluster_descriptions = describe(clusters, cluster_labels=cluster_labels)
    cluster_descriptions.to_csv(c.data.cluster_descriptions, index=False)

    # cluster_info = get_df(c.data.cluster_info)
    # luster_descriptions = get_df(c.data.cluster_descriptions)

    # Create visualization data
    viz_data = cluster_descriptions.merge(right=cluster_info, on=["cluster_label"], how="left")
    viz_data.sort_values(by=["total_social_score", "start_date", "social_score"], ascending=(False, True, False),
                         inplace=True)
    viz_data.to_csv(c.data.cluster_viz_data, index=False)

    visualize(viz_data, save_html=True, html_file_name=c.data.cluster_viz_html, publish=False)


if __name__ == "__main__":
    config_file = "config.json"
    top_n_clusters = 100
    cluster_probability_threshold = 0.7
    main(config_file, top_n_clusters=top_n_clusters, cluster_probability_threshold=cluster_probability_threshold)
