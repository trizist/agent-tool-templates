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

import typing_extensions as te
from markupsafe import Markup


class Markdown(Markup):
    """A class for handling Markdown-formatted text with XML annotations.

    This class extends the Markup class to wrap Markdown content in XML tags for proper
    rendering in the streaming output.

    Parameters
    ----------
    in_obj : str, optional
        The Markdown-formatted text to be wrapped, by default ""
    encoding : str or None, optional
        The encoding to use for the text, by default None
    errors : str, optional
        How to handle encoding errors, by default "strict"

    Returns
    -------
    Markdown
        A new instance of the Markdown class with the content wrapped in XML tags

    Examples
    --------
    >>> md = Markdown("# Header\\nSome text")
    >>> print(md)
    <markdown># Header\nSome text</markdown>
    """

    def __new__(
        cls, in_obj: str = "", encoding: str | None = None, errors: str = "strict"
    ) -> te.Self:
        obj = f"""<markdown>{in_obj}</markdown>"""
        return super().__new__(cls, obj, encoding, errors)
