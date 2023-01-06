from typing import Union, Optional

import chart_studio.plotly as py
import pandas as pd
import plotly.express as px
import plotly.offline

from utils import get_df


def _adjust_description(d: str, cluster_label: str, max_len_chars: int = 60) -> str:
    d = d.replace("$", "USD")
    if len(d) > max_len_chars:
        d = f"{d[:max_len_chars + 3]} …"
    return f"{d.replace('  ', ' ')} ({cluster_label})"


def visualize(path_or_df: Union[str, pd.DataFrame], publish: bool = False, show: bool = True, save_html: bool = True,
              html_file_name: Optional[str] = "metacurate_news_2022.html",
              fig_name: Optional[str] = "metacurate_news_2022") -> None:
    df = get_df(path_or_df)
    df['start'] = pd.to_datetime(df['start_date'])
    df['end'] = pd.to_datetime(df['end_date'])
    df["cluster_probability"] = df["cluster_probability"].apply(lambda x: round(x, 2))
    df["cluster_descriptor"] = df.apply(lambda x: _adjust_description(x["cohere_descriptor"], x["cluster_label"]),
                                        axis=1)
    num_clusters = df.groupby("cluster_label").ngroups
    fig = px.timeline(df.sort_values('total_social_score', ascending=True),
                      title=f"Top {num_clusters} news of 2022",
                      x_start="start",
                      x_end="end",
                      y="cluster_descriptor",
                      hover_data=["title", "cluster_label", "cluster_probability"],
                      color="total_social_score",
                      color_continuous_scale=[(0, "pink"), (0.5, "blue"), (1, "purple")],
                      template="plotly_dark",
                      labels={
                          "social_score": "# shares",
                          "cluster_descriptor": "Cluster description"
                      })
    fig.update_layout(xaxis_title="Date")

    if show:
        fig.show()
    if save_html:
        plotly.offline.plot(fig, filename=html_file_name)
    if publish:
        py.plot(fig, filename=fig_name, auto_open=True)


if __name__ == "__main__":
    data = "data/transient/cluster_viz_data.csv"
    visualize(data, show=False)
