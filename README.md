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

The input data is available in [data/metacurate_news_2022.csv](data/metacurate_news_2022.csv). Example output
is available in [data/output/2022_1/](data/output/2022_1/). The output folder contains:

* A copy of the configuration file used for generating the output, e.g.,
[metacurate_news_2022_1.json](data/output/2022_1/metacurate_news_2022_1.json).
* A copy of the CSV file containing the data used for creating the visualization, e.g.,
[cluster_viz_data.csv](data/output/2022_1/cluster_viz_data.csv).
* A local file containing the visualization as seen in the screenshot above, e.g.,
[metacurate_news_viz_2022.html](data/output/2022_1/metacurate_news_viz_2022.html).
* A list of the top N news clusters, e.g., [README.md](data/output/2022_1/README.md).


## Installation
This section contains instructions for how to install the code, resources, and dependencies
needed to reproduce the clustering of the news headlines as shown in the screenshot above.

### Requirements

* git
* Python (this repo has been tested using Python 3.9)
* pip
* virtualenv
* An API key from Cohere
* Optional: Plotly Chart Studio credentials

### Create and activate a virtual environment

### Clone this repository

### Install dependencies

### Get and set up a Cohere API Key

In order to use [Topically](https://github.com/cohere-ai/sandbox-topically) to describe the clusters,
you need to have an API key from [cohere](https://cohere.ai/). Get an API key by following the instructions in the
Topically repository. Take note of the key, and set the environment variable `COHERE_API_KEY` like so:

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
python src/set_up_plotly_credentials.py
```

to generate and store the credentials. This only has to be done once.

## Run the code

To run the code, simply issue the following:

````bash
python main.py -c configs/metacurate_news_2022_1.json
````

NOTE that this is a long-running process: the vectorization step will take a long time (up to an
hour) if you're running on a CPU, and the clustering takes quite some time too.
