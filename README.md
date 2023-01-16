# metacurate-regularly: clustering of news headlines.

**TL;DR:** This repository contains an experiment for **embedding** and **clustering** news headlines, as well as for
**describing** the resulting clusters, and **plotting** them on a timeline.

The screenshot below shows the output of the clustering exercise: the top 50 news in 2022 regarding AI,
machine learning, data science, and related fields based on data collected by [metacurate.io](https://metacurate.io).
Here is the [live graph](https://chart-studio.plotly.com/~Fredrik/185.embed).

![Top 50 AI/ML/data science news 2022 according to metacurate.io](assets/metacurate_top_50_news_2022.png)



In 2022, my hobby project [metacurate.io](https://metacurate.io) collected 54k+ news items from sources
related to artificial intelligence, machine learning, natural language processing, data science, and other tech
news. This repository contains code for experimenting with clustering headlines, and describing the clusters.

* The data is available in [data/metacurate_news_2022.csv](data/metacurate_news_2022.csv).
* The headlines are embedded using the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
library (see [src/vectorize.py](src/vectorize.py)).
* [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan/) is used for clustering
(see [src/cluster.py](src/cluster.py)).
* The [Topically](https://github.com/cohere-ai/sandbox-topically), and [YAKE](https://github.com/LIAAD/yake) are used
for generating and extracting cluster descriptions, respectively (see [src/describe](src/describe.py)).
* Finally, [Plotly Express](https://github.com/plotly/plotly.py) is used for creating the timeline plot of the
clusters (see [src/visualize.py](src/visualize.py)).


## Installation
This section contains instructions for how to install the code, resources, and dependencies
needed to reproduce the clustering of the news headlines available in
[metacurate_news_2022.csv](data/metacurate_news_2022.csv).

### Requirements

* git
* Python (this repo was developed using Python 3.9)
* pip
* virtualenv
* An API key from Cohere
* Optional: Plotly Chart Studio credentials

### Create and activate a virtual environment

### Clone this repository

### Install dependencies

### Get and set up a Cohere API Key

In order to use [Topically](link) to describe the clusters, you need to have an API key
from Cohere. Get a free API account/key for Cohere here. Take note of the key, and set
the environment variable `COHERE_API_KEY` like so:

```bash
export COHERE_API_KEY=<your_key>
```


### Optional: Get and set up Plotly Chart Studio credentials
In order to publish the generated Plotly plot to the web (Plotly Chart studio), you need to
have an account and set up the credentials locally. Follow the instructions for getting an
account
[here](https://jennifer-banks8585.medium.com/how-to-embed-interactive-plotly-visualizations-on-medium-blogs-710209f93bd)
and edit the file [set_up_plotly_credentials.py](src/set_up_plotly_credentials.py) to include
your `username` and `api_key`.

Run the file:

```bash
python chart_studio.py
```

to generate and store the credentials. This only has to be done once.

## Run the code

To run the code, simply issue the following:

````bash
python main.py
````

NOTE that this is a long-running process: the vectorization step will take a long time (up to an
hour) if you're running on a CPU, and the clustering takes quite some time too.
