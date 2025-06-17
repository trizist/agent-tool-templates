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


def fix_columns_cases(cols_to_fix, cols_to_return):
    if isinstance(cols_to_fix, str):
        cols_to_fix = [cols_to_fix]
    if isinstance(cols_to_return, str):
        cols_to_return = [cols_to_return]
    cols_to_fix = [col.lower() for col in cols_to_fix]
    cols_to_return = {col.lower(): col for col in cols_to_return}
    columns = []
    for col in cols_to_fix:
        if fixed_col := cols_to_return.get(col):
            columns.append(fixed_col)
    return columns
