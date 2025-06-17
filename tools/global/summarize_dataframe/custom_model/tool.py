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
from __future__ import annotations

import pandas as pd
from data_types import Markdown


def summarize_dataframe(
    dataframe: pd.DataFrame,
) -> Markdown:  # recognize a UUID and will inject pandas dataframe
    """Takes in a pandas dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame to summarize

    Returns
    -------
    str
        a markdown formatted table for review
    """
    return Markdown(dataframe.describe().to_markdown())
