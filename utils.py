import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import pandas as pd
from dotmap import DotMap


def from_json(filepath: Path) -> Dict[str, Any]:
    with filepath.open(mode="r") as fh:
        data = json.load(fh)
    return data


def load_config(config_path: Union[str, Path]) -> DotMap:
    if isinstance(config_path, str):
        config_path = Path(config_path)
    return DotMap(from_json(config_path))


def get_logger(name: Optional[str] = None):
    if not name:
        name = "__name__"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_df(path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df, low_memory=False, lineterminator="\n")
    else:
        df = path_or_df
    return df


def get_top_n_cluster_labels(path_or_df: Union[str, pd.DataFrame], top_n_clusters: int,
                             cluster_label_column: str = "cluster_label",
                             social_score_column: str = "social_score") -> List[int]:
    df = get_df(path_or_df)
    df = df.groupby(by=[cluster_label_column])[social_score_column].sum()
    df = df.to_frame().sort_values(by=social_score_column, ascending=False)
    df.reset_index(inplace=True)
    return df["cluster_label"][:top_n_clusters].tolist()
