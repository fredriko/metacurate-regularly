from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from joblib import Memory
from sentence_transformers import SentenceTransformer

from src.utils import get_logger

logger = get_logger("vectorize")


class SentenceTransformerVectorizer:
    def __init__(self, model_name_or_path: str, cache_dir: str) -> None:
        cache = Memory(Path(cache_dir))
        self.vectorize = cache.cache(self.vectorize, ignore=["self"])
        self.model_name_or_path = model_name_or_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name_or_path, device=device)

    def __str__(self) -> str:
        return "SentenceTransformerVectorizer"

    def __repr__(self) -> str:
        return self.__str__()

    def vectorize(
        self, df: pd.DataFrame, text_column: str = "title_normalized"
    ) -> np.ndarray:
        texts = df[text_column].tolist()
        logger.info(
            f"Will vectorize {len(texts)} texts using model {self.model_name_or_path}"
        )
        start_time = time()
        embeddings = self.model.encode(df[text_column].tolist(), show_progress_bar=True)
        logger.info(
            f"Created embeddings from {df.shape[0]} texts in {round(time() - start_time)} seconds"
        )
        return embeddings
