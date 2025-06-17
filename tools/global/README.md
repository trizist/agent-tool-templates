# Global Tools

## Overview

Global Tools are pre-built integrations available through the [Model Registry](https://docs.datarobot.com/en/docs/mlops/deployment/registry/reg-create.html). They can be deployed and used within agents without additional configuration.

## Repository Structure

Each subdirectory contains a standalone custom model implementation for a global tool. The source code is provided (in the custom_model subdirectory) for transparency and can be modified or extended to create custom models tailored to specific use cases.

List of Global Tools:

| **Tool Name**                                                  | **Description**                                                                                                   |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [get_ai_catalog_dataset](./get_ai_catalog_dataset)             | Retrieves datasets from the DataRobot AI Catalog using a `dataset_id`. Returns the dataset in CSV format as raw bytes. |
| [make_ml_predictions](./make_ml_predictions)                   | Performs machine learning predictions using a specified model. Accepts input data and returns prediction results. |
| [make_textgen_predictions](./make_textgen_predictions)         | Generates text predictions based on input data using a text generation model. Suitable for tasks like summarization or text completion. |
| [make_time_series_predictions](./make_time_series_predictions) | Produces time series predictions using a specified model. Useful for forecasting and trend analysis.              |
| [render_chart_plotly](./render_chart_plotly)                   | Creates interactive charts using Plotly based on input data and configuration. Returns a rendered chart object.   |
| [render_chart_vegalight](./render_chart_vegalight)             | Generates a chart from a Vega-Lite specification and returns a JSON with a base64-encoded image of the chart.     |
| [search_ai_catalog](./search_ai_catalog)                       | Searches for datasets in the DataRobot AI Catalog using search terms. Returns matching datasets as a DataFrame.   |
| [summarize_dataframe](./summarize_dataframe)                   | Provides a detailed summary of a pandas DataFrame in Markdown format, including statistics and data insights.     |
