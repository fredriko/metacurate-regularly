"""
Functionality for describing/explaining clusters.
"""
import os
from pathlib import Path
from typing import List, Union, Tuple, Set, Optional

import cohere.error
import pandas as pd
import topically
from sklearn.feature_extraction.text import TfidfVectorizer
from yake import KeywordExtractor

from utils import get_logger, get_df
from TopicallyClient import TopicallyClient

logger = get_logger("Describe")


def compute_tfidf_descriptors(path_or_dataframe: Union[Path, pd.DataFrame],
                              cluster_label_column: str = "cluster_label",
                              text_column: str = "title_normalized",
                              cluster_probability_column: str = "cluster_probability",
                              stop_words: Optional[Set[str]] = None,
                              ngram_range: Tuple[int, int] = (1, 5),
                              min_df: Union[int, float] = 2,
                              max_df: Union[int, float] = 0.7,
                              cluster_probability: Union[None, float] = 0.6,
                              max_num_descriptors: int = 8) -> pd.DataFrame:
    """
    Uses TF-IDF to obtain n-gram descriptions for each cluster in the input file or dataframe.

    :param path_or_dataframe:
    :param cluster_label_column:
    :param text_column:
    :param cluster_probability_column:
    :param stop_words:
    :param ngram_range:
    :param min_df:
    :param max_df:
    :param cluster_probability:
    :param max_num_descriptors:
    :return: A dataframe with two columns: cluster_label, and descriptors, where the latter are terms from the cluster
    that describes it.
    """

    df = get_df(path_or_dataframe)
    logger.info(f"Computing TfIdf descriptors on data frame with {df.shape[0]} rows")

    df_ = _filter_on_cluster_prob(df, cluster_probability_column, cluster_probability)
    df_grouped = _collect_text_from_column(df_, cluster_label_column, "docs", text_column)

    word_vectorizer = _get_tfidf_vectorizer(min_df, max_df, ngram_range, stop_words)

    try:
        vectors = word_vectorizer.fit_transform(df_grouped["docs"]).toarray()
    except ValueError as error:
        logger.warn(f"Encountered error: {error}. Will proceed with different vectorizer values.")
        word_vectorizer = _get_tfidf_vectorizer(1, 1.0, ngram_range, stop_words)
        vectors = word_vectorizer.fit_transform(df_grouped["docs"]).toarray()

    word2idf = dict(zip(word_vectorizer.get_feature_names_out(), word_vectorizer.idf_))
    vocabulary = word_vectorizer.vocabulary_
    reverse_vocabulary = {v: k for k, v in vocabulary.items()}
    idx = vectors.argsort(axis=1)
    tfidf_descriptors = idx[:, -max_num_descriptors:]
    df_grouped["max_descriptors"] = [
        [(reverse_vocabulary.get(item), word2idf.get(reverse_vocabulary.get(item))) for item in row] for row in
        tfidf_descriptors]

    df_grouped["tfidf_descriptors"] = df_grouped["max_descriptors"].apply(_collapse_on_prefixes)
    df_grouped["tfidf_descriptors"] = df_grouped["tfidf_descriptors"].apply(_collapse_on_suffixes)
    df_grouped["tfidf_descriptors"] = df_grouped["tfidf_descriptors"].apply(_clean_up_descriptors)
    df_grouped.drop(columns=["title_normalized", "docs", "max_descriptors"], inplace=True)
    return df_grouped


def _get_tfidf_vectorizer(min_df: Union[int, float], max_df: Union[int, float], ngram_range: Tuple[int, int],
                          stop_words: Set[str]) -> TfidfVectorizer:
    return TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range, use_idf=True,
                           norm="l2", analyzer="word", smooth_idf=True, strip_accents=None,
                           sublinear_tf=True, tokenizer=None, stop_words=stop_words, lowercase=True)


def compute_yake_descriptors(path_or_dataframe: Union[Path, pd.DataFrame],
                             cluster_label_column: str = "cluster_label",
                             text_column: str = "title_normalized",
                             cluster_probability_column: str = "cluster_probability",
                             stop_words: Optional[Set[str]] = None,
                             cluster_probability: Union[None, float] = 0.6,
                             max_num_descriptors: int = 8) -> pd.DataFrame:
    """

    :param path_or_dataframe:
    :param cluster_label_column:
    :param text_column:
    :param cluster_probability_column:
    :param stop_words:
    :param cluster_probability:
    :param max_num_descriptors:
    :return:
    """
    df = get_df(path_or_dataframe)
    logger.info(f"Computing Yake descriptors on data frame with {df.shape[0]} rows")

    df_ = _filter_on_cluster_prob(df, cluster_probability_column, cluster_probability)
    df_grouped = _collect_text_from_column(df_, cluster_label_column, "docs", text_column)

    extractor = KeywordExtractor(stopwords=stop_words, top=max_num_descriptors)
    df_grouped["yake_descriptors"] = df_grouped["docs"].apply(extractor.extract_keywords)
    df_grouped["yake_descriptors"] = df_grouped["yake_descriptors"].apply(_collapse_on_prefixes)
    df_grouped["yake_descriptors"] = df_grouped["yake_descriptors"].apply(_collapse_on_suffixes)
    df_grouped["yake_descriptors"] = df_grouped["yake_descriptors"].apply(_clean_up_descriptors)
    df_grouped.drop(columns=["title_normalized", "docs"], inplace=True)
    return df_grouped


def _collapse_on_prefixes(terms: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """

    :param terms: Tuples of terms and the corresponding IDF
    :return: List of terms sorted according to IDF, higher IDF first which means more informative terms.
    """
    terms.sort(key=lambda item: (len(item[0]), item[0]))
    candidates = []

    def is_prefix_in_list(term, termz) -> bool:
        for tt in termz:
            if tt[0].startswith(term):
                return True
        return False

    for i, term1 in enumerate(terms):
        if not is_prefix_in_list(term1[0], terms[i + 1:]):
            candidates.append(term1)
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _filter_on_cluster_prob(df: pd.DataFrame, cluster_probability_column: str,
                            cluster_probability: float) -> pd.DataFrame:
    # Create a single document from each cluster, filtered by cluster_probability if provided.
    if cluster_probability:
        if cluster_probability_column not in df.columns:
            print(f"Cannot filter on {cluster_probability_column}: not in columns {df.columns}")
            df_ = df.copy()
        else:
            df_ = df[df[cluster_probability_column] >= cluster_probability].copy()
    else:
        df_ = df.copy()
    return df_


def _collect_text_from_column(df: pd.DataFrame, source_column: str, target_column: str, text_column: str) -> \
        pd.DataFrame:
    df_ = df.groupby(source_column)
    df_grouped = df_[text_column].apply(list)
    df_grouped = df_grouped.reset_index()
    df_grouped[target_column] = df_grouped[text_column].apply(lambda x: ". ".join(x))
    return df_grouped


def _collapse_on_suffixes(terms: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    smret = [(term[0][::-1], term[1]) for term in terms]
    collapsed = _collapse_on_prefixes(smret)
    terms_ = [(term[0][::-1], term[1]) for term in collapsed]
    return terms_


def _clean_up_descriptors(terms: List[Tuple[str, float]]) -> List[str]:
    return [term[0] for term in terms]


def compute_cohere_descriptors(path_or_dataframe: Union[Path, pd.DataFrame],
                               cohere_api_key: Optional[str] = None,
                               cluster_label_column: str = "cluster_label",
                               text_column: str = "title_normalized",
                               cluster_probability_column: str = "cluster_probability",
                               cluster_probability: Union[None, float] = 0.75,
                               num_sample_texts: int = 25,
                               num_generations: int = 5) -> Union[None, pd.DataFrame]:
    if not cohere_api_key:
        cohere_api_key = os.environ.get("COHERE_API_KEY", None)
        if not cohere_api_key:
            logger.warn("No API key given for Cohere service. Skipping computing of Cohere cluster descriptors.")
            return None

    app = TopicallyClient(cohere_api_key)
    #app = topically.Topically(cohere_api_key)
    df = get_df(path_or_dataframe)
    logger.info(f"Computing Cohere descriptors on data frame with {df.shape[0]} rows")

    df = df[df[cluster_probability_column] >= cluster_probability]
    logger.info(f"Data frame has {df.shape[0]} rows after thresholding on cluster probability {cluster_probability}")
    df = df[df[cluster_label_column] > -1]
    texts = df[text_column].tolist()
    clusters = df[cluster_label_column].tolist()
    logger.info(f"Will use {len(texts)} texts from which Cohere will select {num_sample_texts} for each cluster")

    topic_names = app.name_topics(texts, clusters, num_generations=num_generations, num_sample_texts=num_sample_texts)
    df_ = None
    if topic_names is not None:
        df_ = pd.DataFrame(
            [{"cluster_label": label, "cohere_descriptor": desc} for label, desc, in topic_names.items()])
        df_.sort_values(by="cluster_label", ascending=True, inplace=True)
    return df_


def describe(path_or_df: Union[Path, pd.DataFrame], cluster_labels: Optional[List[int]] = None,
             cluster_label_column: str = "cluster_label", cluster_probability: float = 0.75) -> pd.DataFrame:
    df = get_df(path_or_df)
    if cluster_labels:
        df = df[df[cluster_label_column].isin(cluster_labels)]
    tfidf_desc_df = compute_tfidf_descriptors(df, cluster_probability=cluster_probability)
    yake_desc_df = compute_yake_descriptors(df, cluster_probability=cluster_probability)
    cohere_desc_df = compute_cohere_descriptors(df, cluster_probability=cluster_probability)
    df_ = tfidf_desc_df.merge(right=yake_desc_df, on=cluster_label_column, how="outer")
    if cohere_desc_df is not None:
        df_ = df_.merge(right=cohere_desc_df, on=cluster_label_column, how="outer")
    return df_
