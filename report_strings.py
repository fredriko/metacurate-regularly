def get_description(top_n: int) -> str:
    description = f"""
In 2022, [metacurate.io](https://metacurate.io) collected 54k+ news items concerning tech, Machine Learning,
Data Science, NLP, Deep Learning, and related areas. The list below was automatically constructed and contains
the top {top_n} stories from the year.

* Each story consists of multiple semantically related news headlines. Each headline is represented by means of
the [sentence-transformer](https://www.sbert.net/) package, and clustered using
[HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html).
* The clusters are ranked based on how many times the news items in them were shared, liked, and commented on
on Facebook. The social interactions were collected for up until a week after each news item was collected
by metacurate.io using [SharedCount](https://www.sharedcount.com/).
* The descriptions of each cluster is created using [Topically](https://github.com/cohere-ai/sandbox-topically).

A GitHub repository with the code for creating the clusters is available at:
https://github.com/fredriko/metacurate-regularly."""
    return description
