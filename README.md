TODO

* Viz: have plotly set width of image, but set height explicitly
* Produce report/list of clusters w top n urls for Medium (markdown?)
* pre-commit, black, linting
* Integrate viz into main.
* Make sure it works end-to-end.
* Refactor code.
* Add command line argument for selecting other config file.

## Set up chart studio

https://jennifer-banks8585.medium.com/how-to-embed-interactive-plotly-visualizations-on-medium-blogs-710209f93bd


```
import chart_studio

username = "<username>"
api_key = "<api_key>"

chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
```