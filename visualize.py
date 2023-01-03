# Plotly
# https://plotly.com/python/gantt/
# https://plotly.com/python-api-reference/generated/plotly.express.timeline.html
# https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
import re

import plotly.express as px
import pandas as pd


def fix_description(d: str, cluster_label: str, max_len_chars: int = 60) -> str:
    d = d.replace("$", "USD")
    if len(d) > max_len_chars:
        d = f"{d[:max_len_chars+3]} …"
    return f"{d.replace('  ', ' ')} ({cluster_label})"


source = pd.read_csv("data/transient/cluster_viz_data.csv", lineterminator="\n")
source['start'] = pd.to_datetime(source['start_date'])
source['end'] = pd.to_datetime(source['end_date'])
source["cluster_probability"] = source["cluster_probability"].apply(lambda x: round(x, 2))
source["cluster_descriptor"] = source.apply(lambda x: fix_description(x["cohere_descriptor"], x["cluster_label"]), axis=1)

fig = px.timeline(source.sort_values('social_score', ascending=True),
                  title="Top N news of 2022",
                  x_start="start",
                  x_end="end",
                  y="cluster_descriptor",
                  hover_data=["title", "cluster_label", "cluster_probability"],
                  color="social_score",
                  color_continuous_scale=[(0, "pink"), (0.5, "blue"), (1, "purple")],
                  template="plotly_dark",
                  labels={
                      "social_score": "# shares",
                      "cluster_descriptor": "Cluster description"
                  })
fig.update_layout(xaxis_title="Date")
fig.show()
