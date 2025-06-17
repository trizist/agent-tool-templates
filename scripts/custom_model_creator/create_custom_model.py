#!/usr/bin/env python3
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

"""
DataRobot Custom Model Creator

This script creates a custom model and its version using the official DataRobot Python client.
It can either create a new custom model or add a new version to an existing model.
"""

import argparse
import os
import sys
import datarobot as dr
from pathlib import Path
import yaml
from typing import Dict, Any
from loguru import logger
import requests
import re


MODEL_METADATA_FILE = 'model-metadata.yaml'

def setup_logging(verbose: bool = False) -> None:
    """Set up loguru logging with nice formatting"""
    logger.remove()

    log_level = "DEBUG" if verbose else "INFO"

    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True
    )


def ensure_api_endpoint(endpoint: str) -> str:
    """Ensure the endpoint ends with /api/v2/"""
    return endpoint.rstrip('/').rstrip('/api/v2') + '/api/v2/'


def parse_resource_bundle_from_yaml_comments(model_dir: Path) -> str | None:
    """Parse YAML file for resource bundle ID in comments"""
    model_metadata_path = model_dir / MODEL_METADATA_FILE
    if not model_metadata_path.exists():
        return None

    try:
        with open(model_metadata_path, 'r') as f:
            content = f.read()

        # Look for comment with format: # suggestedResourceBundleID: <value>
        pattern = r'#\s*suggestedResourceBundleID:\s*(\S+)'
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            resource_bundle_id = match.group(1).strip()
            logger.debug(f"Found suggestedResourceBundleID for this model in {MODEL_METADATA_FILE}: {resource_bundle_id}")
            return resource_bundle_id

        return None

    except Exception as e:
        logger.warning(f"Error parsing {MODEL_METADATA_FILE} for resource bundle ID: {e}")
        return None


def load_model_metadata(model_dir: Path) -> Dict[str, Any]:
    """Load model metadata from model-metadata.yaml or model.yaml file"""
    model_metadata_path = model_dir / MODEL_METADATA_FILE
    if model_metadata_path.exists():
        with open(model_metadata_path, 'r') as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(f"No {MODEL_METADATA_FILE} found in {model_dir}")


def get_environment_id_from_metadata(metadata: Dict[str, Any]) -> str:
    """Extract environment ID from model metadata"""
    environment_id = metadata.get('environmentID')

    if environment_id:
        return environment_id

    raise ValueError(f"Environment ID not found. Expected 'environmentID' in {MODEL_METADATA_FILE}")


def validate_environment_id(environment_id: str) -> str:
    """Validate that the environment ID exists and get its name"""
    try:
        environment = dr.ExecutionEnvironment.get(environment_id)
        return environment.name
    except dr.errors.ClientError as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise ValueError(f"Environment ID {environment_id} not found. Please check the environmentID in your {MODEL_METADATA_FILE} file.")
        else:
            raise ValueError(f"Failed to validate environment ID {environment_id}: {e}")


def validate_existing_model(model_id: str) -> str:
    """Validate that the custom model exists and get its name"""
    try:
        custom_model = dr.CustomInferenceModel.get(model_id)
        return custom_model.name
    except dr.errors.ClientError as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise ValueError(f"Custom model with ID {model_id} not found. Please check the model ID.")
        else:
            raise ValueError(f"Failed to validate custom model ID {model_id}: {e}")


def create_custom_model(metadata: Dict[str, Any]) -> str:
    """Create a custom inference model"""
    if 'name' in metadata and 'targetType' in metadata:
        model_name = metadata.get('name', 'Custom Model')
        model_description = metadata.get('description', 'Custom model created via API')
        metadata_target_type = metadata.get('targetType', 'unstructured')

        inference_model = metadata.get('inferenceModel', {})
        target_name = inference_model.get('targetName')

        target_type_map: dict[str, dr.TARGET_TYPE] = {tt.lower(): tt for tt in dr.TARGET_TYPE.ALL}
        target_type = target_type_map.get(metadata_target_type.lower(), dr.TARGET_TYPE.UNSTRUCTURED)
    else:
        raise ValueError(
            f"Invalid metadata format. Expected 'name', 'targetType', and 'inferenceModel' keys "
            f"in {MODEL_METADATA_FILE}"
        )

    logger.info(f"Creating custom model '{model_name}' with target type '{target_type}'")
    logger.debug(f"Target name: {target_name}")

    try:
        custom_model = dr.CustomInferenceModel.create(
            name=model_name,
            description=model_description,
            target_type=target_type,
            target_name=target_name if target_type != dr.TARGET_TYPE.UNSTRUCTURED else None
        )
        return custom_model.id

    except dr.errors.ClientError as e:
        raise ValueError(f"Failed to create custom model: {e}")


def create_custom_model_version(model_id: str, model_dir: Path, environment_id: str) -> str:
    """Create a custom model version by uploading files"""
    logger.info("Uploading model files and creating version...")

    # Collect all files to upload
    file_paths = []
    for file_path in model_dir.rglob('*'):
        if file_path.is_file():
            file_paths.append(str(file_path))

    logger.debug(f"Found {len(file_paths)} files to upload: {file_paths}")

    try:
        version = dr.CustomModelVersion.create_clean(
            custom_model_id=model_id,
            base_environment_id=environment_id,
            files=file_paths
        )
        return version.id

    except dr.errors.ClientError as e:
        raise ValueError(f"Failed to create custom model version: {e}")


def set_resource_bundle(model_id: str, version_id: str, environment_id: str, resource_bundle_id: str, endpoint: str, api_key: str) -> None:
    """Set resource bundle for custom model version"""
    logger.info(f"Setting resource bundle {resource_bundle_id} for custom model version...")

    try:
        environment = dr.ExecutionEnvironment.get(environment_id)
        environment_version_id = environment.latest_version.id
    except dr.errors.ClientError as e:
        raise ValueError(f"Failed to get environment version ID: {e}")

    # Resource bundles are not available in DR Public API Client, so we will
    # make a requests call to set right bundle
    url = f"{endpoint}customModels/{model_id}/versions/"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        "isMajorUpdate": "false",
        "replicas": 1,  # Default replica count
        "resourceBundleId": resource_bundle_id,
        "baseEnvironmentId": environment_id,
        "baseEnvironmentVersionId": environment_version_id,
    }

    response = requests.patch(url, json=payload, headers=headers)
    if response.status_code not in [200, 201]:
        raise Exception(
            f"Failed to update resource bundle for custom model version: {response.status_code} "
            f"Reason: {response.text}"
        )

    logger.success(f"✓ Set resource bundle {resource_bundle_id} for custom model version {version_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a DataRobot custom model from a local directory or add a version to an existing model'
    )

    parser.add_argument(
        '-e', '--endpoint',
        default=os.getenv('DATAROBOT_ENDPOINT', 'https://app.datarobot.com/'),
        help='DataRobot endpoint URL (default: %(default)s, can be set via DATAROBOT_ENDPOINT env var)'
    )

    parser.add_argument(
        '-a', '--api-key',
        default=os.getenv('DATAROBOT_API_TOKEN'),
        help='DataRobot API key (can be set via DATAROBOT_API_TOKEN env var)'
    )

    parser.add_argument(
        '-m', '--model',
        required=True,
        help='Path to directory containing custom model code (e.g., tools/global/summarize_dataframe)'
    )

    parser.add_argument(
        '-i', '--model-id',
        help='Existing custom model ID to add a new version to (instead of creating a new model)'
    )

    parser.add_argument(
        '-r', '--resource-bundle-id',
        help='Resource bundle ID to set for the custom model version (takes priority over YAML comment)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (debug level)'
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Validate arguments
    if not args.api_key:
        logger.error("API key is required. Use -a/--api-key or set DATAROBOT_API_TOKEN environment variable.")
        sys.exit(1)

    model_dir = Path(args.model)
    if not model_dir.exists():
        logger.error(f"Model directory {model_dir} does not exist.")
        sys.exit(1)

    # Initialize DataRobot client
    endpoint = ensure_api_endpoint(args.endpoint)
    logger.info(f"Using DataRobot endpoint: {endpoint}")

    try:
        # Initialize DataRobot client
        dr.Client(endpoint=endpoint, token=args.api_key)
        logger.debug("DataRobot client initialized successfully")

        # Load model metadata
        logger.info(f"Loading model metadata from {model_dir}...")
        metadata = load_model_metadata(model_dir)

        # Get environment ID from metadata
        logger.info("Getting execution environment from model metadata...")
        environment_id = get_environment_id_from_metadata(metadata)

        # Validate the environment ID exists
        logger.info(f"Validating execution environment {environment_id}...")
        env_name = validate_environment_id(environment_id)
        logger.info(f"Using execution environment: {env_name} ({environment_id})")

        # Determine resource bundle ID (CLI argument takes priority over YAML comment)
        resource_bundle_id = args.resource_bundle_id
        if not resource_bundle_id:
            resource_bundle_id = parse_resource_bundle_from_yaml_comments(model_dir)

        if resource_bundle_id:
            logger.info(f"Will set resource bundle ID: {resource_bundle_id}")

        # Determine if we're creating a new model or adding to existing
        if args.model_id:
            # Validate existing model
            logger.info(f"Validating existing custom model {args.model_id}...")
            model_name = validate_existing_model(args.model_id)
            logger.info(f"Adding new version to existing model: {model_name} ({args.model_id})")
            model_id = args.model_id
        else:
            # Create new custom model
            logger.info("Creating new custom model...")
            model_id = create_custom_model(metadata)
            logger.success(f"✓ Created custom model with ID: {model_id}")

        # Create custom model version
        logger.info("Creating custom model version...")
        version_id = create_custom_model_version(model_id, model_dir, environment_id)
        logger.success(f"✓ Created custom model version with ID: {version_id}")

        # Set resource bundle if specified
        if resource_bundle_id:
            set_resource_bundle(model_id, version_id, environment_id, resource_bundle_id, endpoint, args.api_key)

        # Success message
        if args.model_id:
            logger.success("🎉 Successfully added new version to existing custom model!")
        else:
            logger.success("🎉 Successfully created custom model!")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Version ID: {version_id}")
        logger.info(f"You can view it in the DataRobot UI at: {endpoint.rstrip('/api/v2/')}/registry/custom-model-workshop/{model_id}/assemble")
        logger.info(f"When applicable, please fill in the runtime parameters and adjust the model resource bundle")

    except dr.errors.ClientError as e:
        logger.error(f"DataRobot API Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 
