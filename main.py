import argparse
import warnings

from dotmap import DotMap

from src.cluster import cluster
from src.data import (
    normalize_data,
    sort_filter_clusters,
    compute_cluster_info,
    prep_directory_structure,
    create_viz_data,
    copy_config,
)
from src.describe import describe
from src.report import report
from src.utils import get_logger, load_config, get_top_n_cluster_labels
from src.vectorize import SentenceTransformerVectorizer
from src.visualize import visualize

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = get_logger("main")


def main(config_file: str) -> None:
    c: DotMap = load_config(config_file)
    logger.info(f"Got config: {c}")

    describe_n = (
        c.params.report_top_n
        if c.params.report_top_n > c.params.visualize_top_n
        else c.params.visualize_top_n
    )

    prep_directory_structure(c)
    copy_config(c, config_file)

    # Normalize data
    df = normalize_data(c.data.raw, c.resources.omit_strings)
    df.to_csv(c.data.normalized, index=False)

    # Vectorize
    logger.info(f"Got {df.shape[0]} titles from file: {c.data.normalized}.")
    df = df[df["social_score"] > 0]
    logger.info(
        f"Got {df.shape[0]} titles after keeping only those with a social score over 0."
    )
    vectorizer = SentenceTransformerVectorizer(
        c.vectorizer.model_name_or_path, c.data.cache
    )
    embeddings = vectorizer.vectorize(df)

    # Cluster
    clusters = cluster(embeddings, df, sort=True, **c.clusterer.toDict())
    clusters.to_csv(c.data.clustered, index=False)

    # Post-process clusters
    clusters = sort_filter_clusters(
        clusters, cluster_probability=c.params.cluster_probability
    )
    clusters.to_csv(c.data.clustered, index=False)
    cluster_info = compute_cluster_info(clusters)
    cluster_info.to_csv(c.data.cluster_info, index=False)

    # Describe
    cluster_labels = get_top_n_cluster_labels(cluster_info, top_n_clusters=describe_n)
    cluster_descriptions = describe(
        clusters,
        cluster_labels=cluster_labels,
        cluster_probability=c.params.cluster_probability,
    )
    cluster_descriptions.to_csv(c.data.cluster_descriptions, index=False)

    # Create visualization data
    viz_data = create_viz_data(cluster_descriptions, cluster_info)
    viz_data.to_csv(c.data.cluster_viz_data, index=False)

    visualize(
        viz_data,
        visualize_top_n_clusters=c.params.visualize_top_n,
        save_html=True,
        html_file_name=c.data.cluster_viz_html,
        plotly_file_name=c.params.plotly_file_name,
        publish=c.params.publish_to_plotly,
        title=c.params.title,
    )

    report(
        viz_data,
        c.data.cluster_report,
        title=c.params.title,
        top_n_clusters=c.params.report_top_n,
    )


def setup_argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c",
        "--config",
        type=str,
        help="Specify the configuration file to use.",
        required=True,
    )
    return p


if __name__ == "__main__":
    main(setup_argparse().parse_args().config)
