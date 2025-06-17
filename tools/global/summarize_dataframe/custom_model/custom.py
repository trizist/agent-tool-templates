# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from io import StringIO
from typing import Union

import pandas as pd
import tool
from opentelemetry.instrumentation.requests import RequestsInstrumentor

instrument_requests = RequestsInstrumentor().instrument()


def load_model(input_dir: str):
    """This is called when the model is loaded by DataRobot.
    Custom model hook for loading our the model or artifacts for use by scoring code.
    """
    _ = input_dir
    return "model"


def score_unstructured(model, data: Union[bytes, str], **kwargs):
    """
    This is the main scoring hook invoked by DataRobot during scoring of the unstructured model.

    Args:
        model: Loaded model from load_model() hook.
        data: Incoming JSON data containing the dataframe to be summarized in the "dataframe" key
        kwargs: Additional keyword arguments.

    Returns:
        JSON response with search results
    """

    request = json.loads(data)
    # authorization_context for this tool is not neede because it only calls DataRobot API
    payload = request.get("payload", {})

    datafile = StringIO(payload["dataframe"])
    data_df = pd.read_csv(datafile)

    result = tool.summarize_dataframe(data_df)
    response = {"result": result}

    return json.dumps(response), {"mimetype": "application/json", "charset": "utf-8"}

