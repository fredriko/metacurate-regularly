import re
from pathlib import Path
from typing import Optional, Union, Any

import pandas as pd
from dotmap import DotMap
from tqdm import tqdm

from utils import get_logger, get_df

logger = get_logger("data")


def load_omit_strings(omit_strings_file: str, string_column: str = "term") -> list[str]:
    """
    Reads a CSV file with strings to omit from web page titles.

    :param omit_strings_file: The file with the strings to omit.
    :param string_column:
    :return: A list of strings to omit, sorted according to string length.
    """
    if omit_strings_file is None:
        return []
    df = pd.read_csv(omit_strings_file)
    df["length"] = df[string_column].str.len()
    df.sort_values(by="length", ascending=False, inplace=True)
    return df[string_column].to_list()


def normalize_title(
    title: str,
    omit_strings: list[str],
    lower_case: bool = False,
    unaffected_titles: Optional[list[str]] = None,
) -> str:
    """
    Removes unwanted substrings from a web page title.

    :param title: The title from which to remove unwanted substrings.
    :param omit_strings: The unwanted strings to remove.
    :param lower_case: Set to True if the returned normalized title should be lower-cased. Set to False otherwise.
    :param unaffected_titles: List in which to store all titles that were not affected by the attempt to remove
    unwanted substrings. Mainly for diagnostic purposes as the unaffected_titles list can be used to inspect titles
    for more patterns to add to the omit list.
    :return: The normalized title.
    """
    title = re.sub("\r", "", title)
    title = re.sub(" ", " ", title)
    title = re.sub("", " ", title)
    title = re.sub("", " ", title)
    title = re.sub("\n+", " ", title).strip()

    original_title = title
    for ngram in omit_strings:
        before = len(title)
        title = title.replace(ngram, "")
        after = len(title)
        if after != before:
            break
    if unaffected_titles is not None and len(original_title) == len(title):
        unaffected_titles.append(original_title)
    return title.strip().lower() if lower_case else title.strip()


def normalize_title_date(
    data_file: str,
    omit_strings_file: str,
    title_column: str = "title",
    date_column: str = "listed_at_date",
    **kwargs,
) -> pd.DataFrame:
    logger.info(f"Reading URL data from file: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"Got {df.shape[0]} URLs.")
    logger.info(f"Reading omit strings from file: {omit_strings_file}")
    omit_strings = load_omit_strings(omit_strings_file)
    tqdm.pandas()
    logger.info("Normalizing titles...")
    df["title_normalized"] = df[title_column].progress_apply(
        lambda x: normalize_title(x, omit_strings, **kwargs)
    )
    df.insert(loc=1, column="title_normalized", value=df.pop("title_normalized"))

    logger.info("Normalizing dates...")
    df["date_normalized"] = df[date_column].progress_apply(
        lambda x: pd.Timestamp(x).replace(hour=0, minute=0, second=0)
    )
    return df


def prep_output_directory(config: DotMap) -> None:
    normalized_data_dir = Path(config.data.normalized).parent
    if not normalized_data_dir.exists():
        logger.info(f"Creating output directory: {normalized_data_dir.resolve()}")
        normalized_data_dir.mkdir(parents=True)


def sort_filter_clusters(
    path_or_df: Union[str, pd.DataFrame],
    cluster_label_column: str = "cluster_label",
    cluster_probability_column: str = "cluster_probability",
    normalized_date_column: str = "date_normalized",
    social_score_column: str = "social_score",
    cluster_probability: float = 0.75,
) -> pd.DataFrame:
    df = get_df(path_or_df)
    logger.info(f"Sorting and filtering data frame with {df.shape[0]} rows")
    df = df[df[cluster_probability_column] >= cluster_probability]
    df = df.sort_values(
        by=[
            cluster_label_column,
            normalized_date_column,
            social_score_column,
            cluster_probability_column,
        ],
        ascending=(False, False, False, False),
    )
    return df


def compute_cluster_info(
    path_or_df: Union[str, pd.DataFrame],
    cluster_label_column: str = "cluster_label",
    social_score_column: str = "social_score",
) -> pd.DataFrame:
    df = get_df(path_or_df)
    gf = df.groupby(by=[cluster_label_column])
    df_contents: list[dict[str, Any]] = []
    for g in gf:
        total_social_score = g[1][social_score_column].sum()
        for i, row in g[1].iterrows():
            df_contents.append(
                {
                    cluster_label_column: g[0],
                    "cluster_probability": row["cluster_probability"],
                    social_score_column: row[social_score_column],
                    "total_social_score": total_social_score,
                    "start_date": row["date_normalized"],
                    "end_date": pd.Timestamp(row["date_normalized"])
                    + pd.Timedelta(days=1),
                    "url": row["url"],
                    "title": row["title"],
                }
            )
    df_ = pd.DataFrame(df_contents)
    df_.sort_values(by=["total_social_score"], ascending=False, inplace=True)
    return df_


def create_viz_data(
    descriptions_path_or_df: Union[str, pd.DataFrame],
    info_path_or_df: Union[str, pd.DataFrame],
    cluster_label_column: str = "cluster_label",
    total_social_score_column: str = "total_social_score",
    start_date_column: str = "start_date",
    social_score_column: str = "social_score",
) -> pd.DataFrame:

    cluster_descriptions = get_df(descriptions_path_or_df)
    cluster_info = get_df(info_path_or_df)
    viz_data = cluster_descriptions.merge(
        right=cluster_info, on=[cluster_label_column], how="left"
    )
    viz_data.sort_values(
        by=[total_social_score_column, start_date_column, social_score_column],
        ascending=(False, True, False),
        inplace=True,
    )
    return viz_data
