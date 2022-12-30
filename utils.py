import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

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


def get_df(path_or_df: Union[Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(path_or_df, Path):
        df = pd.read_csv(path_or_df, lineterminator="\n")
    else:
        df = path_or_df
    return df
