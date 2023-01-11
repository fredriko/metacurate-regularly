"""
Functionality for creating a textual report of the visualization data.
"""
import datetime
import warnings
from pathlib import Path
from typing import Union, Optional

import pandas as pd
from w3lib.url import canonicalize_url, url_query_cleaner

from src.utils import get_df, get_logger, get_top_n_cluster_labels

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = get_logger("report")


def report(
    path_or_df: Union[str, pd.DataFrame],
    report_file: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    cluster_label_column: str = "cluster_label",
    total_social_score_column: str = "total_social_score",
    url_column: str = "url",
    title_column: str = "title",
    start_date_column: str = "start_date",
    descriptor_column: str = "cohere_descriptor",
    top_n_clusters: int = 100,
) -> None:
    df = get_df(path_or_df)

    top_n: list[int] = get_top_n_cluster_labels(df, top_n_clusters)
    df = df[df[cluster_label_column].isin(top_n)]
    df = df.sort_values(
        by=[total_social_score_column, start_date_column],
        ascending=(False, True),
    )
    rank = 1

    logger.info(f"Saving report to file: {report_file}")
    report_file = Path(report_file)
    with report_file.open(mode="w") as fh:
        if title:
            fh.write(f"# {title}\n")
        if description:
            fh.write(f"{description}\n")
        for cluster_label, label_df in df.groupby(
            by=[cluster_label_column], sort=False
        ):
            description = label_df[descriptor_column].tolist()[0]
            stories = []
            for index, row in label_df.iterrows():
                title = row[title_column]
                url = row[url_column]
                uu = canonicalize_url(url_query_cleaner(url, [], remove=False))
                dt = datetime.datetime.strftime(
                    pd.Timestamp(row[start_date_column]).date(), "%Y-%m-%d"
                )
                stories.append(f"* {dt} - [{title}]({uu})\n")

            header = (
                f"<details><summary>{rank}: {description} ({label_df.shape[0]} items)</summary><p>\n\n"
                f"### {rank}: {description}\n"
            )
            body = "".join(stories)
            footer = "\n</p></details>\n"

            fh.write(f"{header}{body}{footer}")

            rank += 1
