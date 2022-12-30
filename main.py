from pathlib import Path

import pandas as pd

from utils import get_logger, load_config
from data import normalize_title_date
from vectorize import SentenceTransformerVectorizer
from dotmap import DotMap
from cluster import HdbScanClusterer

logger = get_logger("main")

if __name__ == "__main__":
    config_file = Path("config.json")
    c: DotMap = load_config(config_file)
    normalize_title_date(c.data.raw, c.resources.omit_strings, c.data.normalized)

    df = pd.read_csv(c.data.normalized, lineterminator='\n', nrows=None)
    vectorizer = SentenceTransformerVectorizer(c.vectorizer.model_name_or_path, c.data.cache)
    embeddings = vectorizer.vectorize(df)

    clusterer = HdbScanClusterer()
    clusters = clusterer.cluster(embeddings[:20000], df[:20000], sort=True, **c.clusterer.toDict())
    clusters.to_csv(c.data.clustered, index=False)


