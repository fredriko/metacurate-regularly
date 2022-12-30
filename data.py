from pathlib import Path
from typing import List, Optional
import re
import pandas as pd
from utils import get_logger
from tqdm import tqdm

logger = get_logger("data")


def load_omit_strings(omit_strings_file: Path, string_column: str = "term") -> List[str]:
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


def normalize_title(title: str, omit_strings: List[str], lower_case: bool = False,
                    unaffected_titles: Optional[List[str]] = None) -> str:
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


def normalize_title_date(data_file: Path, omit_strings_file: Path, normalized_data_file: Path,
                         title_column: str = "title", date_column: str = "listed_at_date", **kwargs) -> None:
    logger.info(f"Reading URL data from file: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"Got {df.shape[0]} URLs.")
    logger.info(f"Reading omit strings from file: {omit_strings_file}")
    omit_strings = load_omit_strings(omit_strings_file)
    tqdm.pandas()
    logger.info("Normalizing titles...")
    df["title_normalized"] = df[title_column].progress_apply(lambda x: normalize_title(x, omit_strings, **kwargs))
    df.insert(loc=1, column="title_normalized", value=df.pop("title_normalized"))

    logger.info("Normalizing dates...")
    df["date_normalized"] = df[date_column].progress_apply(
        lambda x: pd.Timestamp(x).replace(hour=0, minute=0, second=0))
    logger.info(f"Saving normalized URL data to file: {normalized_data_file}")
    df.to_csv(normalized_data_file, index=False)
