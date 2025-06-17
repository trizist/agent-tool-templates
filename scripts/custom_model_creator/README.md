# DataRobot Custom Model Creator

This script creates a custom model and its version in DataRobot using the official DataRobot Python client. It takes a directory containing custom model code (like those in `tools/global/`) and uploads it to DataRobot.

## Prerequisites

1. Python 3.11 or higher
2. DataRobot API access with a valid API token
3. Install requirements with `pip install -r requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic usage:

```bash
python create_custom_model.py -m PATH_TO_CUSTOM_MODEL -a YOUR_API_TOKEN
```

### Using environment variables:

```bash
export DATAROBOT_API_TOKEN="your-api-token"
export DATAROBOT_ENDPOINT="https://app.datarobot.com/"
python create_custom_model.py -m PATH_TO_CUSTOM_MODEL
```

## Expected Directory Structure

The script expects the model directory to have the following structure:

```
 custom_model/
 ├── model-metadata.yaml  # Model metadata
 ├── custom.py            # Main model code
 ├── requirements.txt     # Python dependencies
 └── ... other files
```

## Metadata File (`model-metadata.yaml`) Format

See [DataRobot documentation](https://docs.datarobot.com/en/docs/modeling/special-workflows/cml/cml-ref/cml-validation.html) for the format description, or refer to examples in this repository.

## Output

Upon successful execution, the script will output:
- The created custom model ID
- The created custom model version ID
- A URL to view the model in the DataRobot UI
