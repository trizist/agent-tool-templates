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
import logging
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """Configuration class for the tool with IDE autocompletion support."""

    # Optional: OpenAI API key to use for making prediction explanations.
    # When using DataRobot deployment, use DataRobot API key instead.
    OPENAI_API_KEY: str = field(default="")
    # Optional: OpenAI endpoint to use for making prediction explanations.
    # To use DataRobot deployment, use "{datarobot_url}/api/v2/deployments/{deployment_id}"
    OPENAI_ENDPOINT: str = field(default="")
    # Optional: Azure OpenAI API version to use for making prediction explanations.
    # "2024-08-01-preview" or newer is required for the predictions explanations to work.
    OPENAI_API_VERSION: str = field(default="")
    DATAROBOT_ENDPOINT: str = field(default="")
    DATAROBOT_API_TOKEN: str = field(default="")

    @classmethod
    def parse_mlops_param(cls, param: str) -> Any:
        """Parse MLOPS_RUNTIME_PARAM value."""
        try:
            param = json.loads(param)
            if isinstance(param, dict):
                if param.get("type") == "string":
                    return param["payload"]
                if len(param) == 1:
                    return list(param.values())[0]
                elif "payload" in param:
                    payload = param["payload"]
                    if "apiToken" in payload:
                        return payload["apiToken"]
            return param
        except json.JSONDecodeError:
            return param
        except TypeError:
            return None

    @classmethod
    def load(cls) -> "ToolConfig":
        """
        Load configuration with the following precedence:
        1. MLOPS_RUNTIME_PARAM_{name}
        2. Environment variable {name}
        3. Default value
        """
        # Create a dict to store all config values
        final_config: Dict[str, Any] = {}

        # Get all field names from the dataclass
        fields = cls.__dataclass_fields__.keys()

        for key in fields:
            # Check MLOPS_RUNTIME_PARAM first
            rt_name = f"MLOPS_RUNTIME_PARAM_{key}"
            rt_value = os.getenv(rt_name)
            if rt_value is not None:
                final_config[key] = cls.parse_mlops_param(rt_value)
                continue

            # Then check regular environment variables
            env_value = os.getenv(key)
            if env_value is not None:
                final_config[key] = env_value
                continue

        return cls(**final_config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration values."""
        return self.get(key)


# Create a global config instance
config = ToolConfig.load()


def getenv(name: str, default: Optional[str] = None) -> str:
    """
    Backward-compatible getenv function that uses the new config class.
    """
    return config.get(name, default)
