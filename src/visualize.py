from typing import Union, Optional

import chart_studio.plotly as py
import pandas as pd
import plotly.express as px
import plotly.offline

from src.utils import get_df, get_top_n_cluster_labels, get_logger

logger = get_logger("visualize")


def _adjust_description(d: str, cluster_label: str, max_len_chars: int = 70) -> str:
    d = d.replace("$", "USD")
    if len(d) > max_len_chars:
        d = f"{d[:max_len_chars + 3]}…"
    return f"{d.replace('  ', ' ')} ({cluster_label})"


def visualize(
    path_or_df: Union[str, pd.DataFrame],
    visualize_top_n_clusters: int = 50,
    publish: bool = False,
    show: bool = True,
    save_html: bool = True,
    html_file_name: Optional[str] = "visualization.html",
    plotly_file_name: Optional[str] = "visualization",
    height: Optional[int] = None,
    width: Optional[int] = None,
    title: Optional[str] = None,
) -> None:

    df = get_df(path_or_df)
    df["start"] = pd.to_datetime(df["start_date"])
    df["end"] = pd.to_datetime(df["end_date"])
    df["cluster_probability"] = df["cluster_probability"].apply(lambda x: round(x, 2))
    df["cluster_descriptor"] = df.apply(
        lambda x: _adjust_description(x["cohere_descriptor"], x["cluster_label"]),
        axis=1,
    )
    top_n: list[int] = get_top_n_cluster_labels(df, visualize_top_n_clusters)
    df_viz = df[df["cluster_label"].isin(top_n)]

    if title is None:
        title = f"Top {df_viz['cluster_label'].nunique()} stories"
    fig = px.timeline(
        df_viz.sort_values("total_social_score", ascending=True),
        title=title,
        x_start="start",
        x_end="end",
        y="cluster_descriptor",
        hover_data=["title", "cluster_label", "cluster_probability"],
        color="total_social_score",
        color_continuous_scale=[(0, "white"), (0.3, "yellow"), (1, "orange")],
        template="plotly_dark",
        height=height,
        width=width,
        labels={
            "total_social_score": "# shares",
            "cluster_descriptor": "Cluster description (cluster id)",
        },
    )
    fig.update_layout(xaxis_title="Date")

    if show:
        fig.show()
    if save_html:
        logger.info(f"Saving visualization to file: {html_file_name}")
        plotly.offline.plot(fig, filename=html_file_name)
    if publish:
        logger.info(f"Publishing figure named '{plotly_file_name}' to Plotly")
        py.plot(fig, filename=plotly_file_name, auto_open=True)
