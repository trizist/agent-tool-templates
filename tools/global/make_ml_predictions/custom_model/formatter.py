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
import functools
import json
import logging
import re
from time import sleep
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import datarobot as dr
import pandas as pd

logger = logging.getLogger(__name__)


@functools.cache
def get_important_features(deployment_id: str) -> List[str]:
    """Gets up to the top 10
    most important features of a model based on SHAP impact.
    Parameters
    ----------
    deployment_id : str
        the datarobot deployment to obtain.
    Returns
    -------
    List[str]
        names of features as a list
    """
    c = dr.client.Client()
    depl = c.get(f"deployments/{deployment_id}").json()
    model_id = depl["model"].get("id")
    shap_params = {"source": "training", "entityId": model_id, "quickCompute": True}

    try:
        resp = c.post("insights/shapImpact", data=shap_params)
    except dr.errors.ClientError as e:
        if "404" in str(e):
            logger.warning(f"Model {model_id} does not have SHAP Impact. Trying Feature Impact.")
            resp = c.get(f"insights/featureImpact/models/{model_id}")

    if status_location := resp.headers.get("Location"):
        job_id = status_location.split("/")[-2]
        job_running = True
        while job_running:
            status = c.get(f"status/{job_id}").json()
            if status.get("status") == "ERROR":  # likely not shap
                resp = c.get(f"insights/featureImpact/models/{model_id}")
                impacts = resp.json()["data"][0]["data"]["featureImpacts"]
                job_running = False
            elif status.get("data", False):
                raw_impacts = status.get("data")
                impacts = raw_impacts[0]["data"]["shapImpacts"]
                job_running = False
            else:
                sleep(3)
    else:
        raw_impacts = json.loads(resp.text)
        impacts = raw_impacts["data"][0]["data"]["featureImpacts"]
    ranked_features = list(
        map(
            lambda r: r.get("featureName"),
            reversed(sorted(impacts, key=lambda x: x.get("impactNormalized"))),
        )
    )
    if len(ranked_features) > 10:
        return ranked_features[0:10]
    else:
        return ranked_features


class PredictionFormatter:
    """
    Format DataRobot deployment predictions.
    Reshape predictions from DataRobot RT and Batch deployment prediction API
    into better format
    """

    preds_df: pd.DataFrame
    deployment: dr.models.deployment.Deployment
    settings: Dict[str, Any]
    explanation_columns: List[str] = []

    def __init__(
        self,
        preds: pd.DataFrame,
        deployment: dr.models.deployment.Deployment,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.preds_df = preds
        self.deployment = deployment
        self.all_prediction_columns = list(
            filter(lambda col: re.search(re.escape("_PREDICTION"), col), preds.columns)
        )
        self.settings = deployment.get_predictions_by_forecast_date_settings()
        self.explanation_columns = list(
            filter(lambda col: re.search(re.escape("EXPLANATION"), col), preds.columns)
        )
        self.total_expls = len(self.explanation_columns) / 4.0
        self.model_info = model_info

    def format(self) -> pd.DataFrame:
        isStructured = not self.deployment.model["unstructured_model_kind"]
        if date_col := self.settings.get("column_name"):
            self.preds_df.index = self.preds_df[date_col.replace(" (actual)", "")]
        if (not self.deployment.model["unsupervised_mode"]) and isStructured:
            df = self.preds_df.rename(
                columns={self.deployment.model["target_name"] + "_PREDICTION": "prediction"}
            )
        else:
            df = self.preds_df.rename(columns={"PREDICTION": "prediction"})

        if self.deployment.model["unsupervised_mode"] and isStructured:
            df = self._process_anomaly_detection_results(df)
        if self.deployment.model["unsupervised_mode"] and isStructured:
            df = self._process_clustering_results(df)
        if self.settings["enabled"]:
            df = self._process_timeseries_results(df)
        if self.deployment.model["target_type"] == "Multilabel":
            df = self._process_multilabel_results(df)
        if self.deployment.model["target_type"] == "Binary":
            df = self._process_binary_predictions(df)
        df = self._process_supervised_model_class_probs(df)

        # Universal changes for all models
        df.drop(
            [
                "DEPLOYMENT_APPROVAL_STATUS",
                self.deployment.model["target_name"],
                "CLASS_1",
            ],
            axis=1,
            errors="ignore",
            inplace=True,
        )
        if len(self.explanation_columns) > 0:
            pass
            # df = self._process_rt_pred_explanations(df)

        return df

    def _process_rt_pred_explanations(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            features_ranked = get_important_features(deployment_id=self.deployment.id)
        except Exception as e:
            logger.warning(f"""Failed to Get Feature importance not adding Expls""")
            return df
        result_dict = {}
        for feature in features_ranked:
            result_dict[feature] = []
        for _, row in df.iterrows():
            # Initialize feature values as None
            feature_values = {feature: None for feature in features_ranked}

            # Look through explanation columns
            for i in range(1, int(self.total_expls)):  # 6 explanation columns
                feature_name = row[f"EXPLANATION_{i}_FEATURE_NAME"]
                if feature_name in features_ranked:
                    feature_values[feature_name] = row[f"EXPLANATION_{i}_QUALITATIVE_STRENGTH"]

            # Add found values to result
            for feature in features_ranked:
                result_dict[feature].append(feature_values[feature])

        mapper = {
            "+": 1,
            "++": 2,
            "+++": 3,
            "-": -1,
            "--": -2,
            "---": -3,
        }
        return pd.concat(
            (
                df.drop(columns=self.explanation_columns),
                pd.DataFrame(result_dict, index=df.index),
            ),
            axis=1,
        ).applymap(lambda r: mapper.get(r, r) if r is not None else r)

    def _process_supervised_model_class_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        renamer = {
            old_col: PredictionFormatter._standardize_prediction_label(
                self.deployment.model["target_name"], old_col
            )
            for old_col in self.all_prediction_columns
        }
        df = df.rename(columns=renamer)
        df = df.drop(
            filter(lambda col: re.search(re.escape("_PREDICTION"), col), df.columns),
            axis=1,
            errors="ignore",
        )
        return df

    def _process_binary_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["THRESHOLD", "POSITIVE_CLASS"], axis=1, errors="ignore")
        return df

    def _process_multilabel_results(self, df: pd.DataFrame) -> pd.DataFrame:
        def swap_target(in_str: str) -> str:
            return in_str.replace("target_", "class_").replace("_PROBABILITY", "")

        def col_finder(cols: List[str], finder: str) -> List[str]:
            """Multi label batch predictions always return DECISION, THRESHOLD, and PROBABILITY
            if we ask for class_probabilities we want the PROBABILITY column if we don't we want the DECISION
            column. We want to keep all the other columns.
            """
            return [
                col
                for col in cols
                if (finder.lower() in col.lower())
                and (
                    ("THRESHOLD" not in col.lower())
                    or ("DECISION" not in col.lower())
                    or ("PROBABILITY" not in col.lower())
                )
            ]

        cols_to_keep = col_finder(df.columns.to_list(), "PROBABILITY")
        renamer = {col: swap_target(col) for col in cols_to_keep}
        new_cols = [swap_target(col) for col in cols_to_keep]
        df = df.rename(columns=renamer)[new_cols]
        return df

    def _process_timeseries_results(self, df: pd.DataFrame) -> pd.DataFrame:
        renamer = {
            "FORECAST_POINT": "forecastPoint",
            "FORECAST_DISTANCE": "forecastDistance",
        }

        try:
            series_col = self.model_info.timeseries["series_column_name"].replace(" (actual)", "")
            renamer[series_col] = "seriesId"
        except KeyError as e:
            logger.error(f"Error finding series column: {e}")
        except Exception as e:
            logger.error(f"Other error processing series column: {e}")

        try:
            date_col = self.model_info.timeseries["datetime_column_name"].replace(" (actual)", "")
            df[date_col] = pd.to_datetime(df[date_col])
            renamer[date_col] = "timestamp"
        except KeyError as e:
            logger.error(f"No datetime column name found in model info: {e}")
        except Exception as e:
            logger.error(f"Error converting datetime column: {e}")

        df = df.rename(columns=renamer)
        return df

    def _process_clustering_results(self, df: pd.DataFrame) -> pd.DataFrame:
        if "anomaly_score" not in df.columns and "ANOMALY_SCORE" not in df.columns:
            renamer = {
                old_col: PredictionFormatter._standardize_clustering_label(old_col)
                for old_col in self.all_prediction_columns
            }
            df = df.rename(columns=renamer)
            self.all_prediction_columns = [value for _, value in renamer.items()]
            return df
        else:
            return df

    def _process_anomaly_detection_results(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ANOMALY_SCORE" in df.columns:
            df["prediction"] = (df["ANOMALY_SCORE"] > 0.5).replace(
                {True: "Anomalous", False: "Normal"}
            )
            df = df.rename(columns={"ANOMALY_SCORE": "anomaly_score"})
            df = df.drop("Anomaly", axis=1, errors="ignore")
            df = df.assign(
                class_Anomalous=lambda df: df["anomaly_score"],
                class_Normal=lambda df: 1 - df["anomaly_score"],
            ).drop("anomaly_score", axis=1)
            return df
        else:
            return df

    def _process_batch_explanations(self, df: pd.DataFrame) -> pd.DataFrame:
        expl_columns = [col for col in df.columns if "EXPLANATION" in col]

        renamer = {
            old_col: PredictionFormatter._standardize_explanation_label(old_col)
            for old_col in expl_columns
        }
        df = df.rename(columns=renamer)
        return df

    @staticmethod
    def _standardize_clustering_label(column_label: str) -> str:
        """Batch Predictions returns "Cluster 2_PREDICTION" but we want "class_Cluster 2" for consistency.
        Args:
            column_label (str): _description_
        Returns
        -------
            str: New coumn name
        """
        return f"class_{column_label.replace('_PREDICTION', '')}"

    @staticmethod
    def _standardize_prediction_label(incol: str, column_label: str) -> str:
        """Standardize class prediction column names.
        Batch predictions have a funny way of naming columns in Multiclass models.
        Say you are predicting the weather, the column names will be:
        - Weather_Rain_PREDICTION
        - Weather_Snow_PREDICTION
        - Weather_SUN_PREDICTION
        This function converts that format to be:
        - Weather_Rain_PREDICTION -> class_Rain_prediction
        - Weather_Snow_PREDICTION -> class_Snow_prediction
        - Weather_SUN_PREDICTION -> class_SUN_prediction
        This function illustrates why this is needed so badly. To infer the class names from the results
        of batch prediction, you need to REGEX with the target name to extract the middle value.
        a data scientist using DRX would only need to do:
        ```
        >>> prediction = 'class_Rain_prediction'
        >>> class_name = prediction[6:len(prediction)- 11
        >>> print(class_name)
        Rain
        ```
        Parameters
        ----------
        incol : str
            The target column this will be used to shape the new column name
        column_label : str)
            The column name being converted
        Returns
        -------
        str
            The new column name
        """
        m = re.match(re.escape(incol + "_"), column_label)
        s = re.search(re.escape("_PREDICTION"), column_label)
        if s:
            return f"class_{column_label[m.end() : s.start()]}"  # type: ignore[union-attr]
        else:
            return column_label

    @staticmethod
    def _standardize_explanation_label(column_label: str) -> str:
        """Standardize explanation column names.
        Batch predictions have a funny way of returning columns with the prediction
        explanation value and information. Consider a model that is predicting the weather (sunny, rainy, or snowy):
        - Weather_EXPLANATION_1_STRENGTH
        - Weather_EXPLANATION_1_ACTUAL_VALUE
        - Weather_EXPLANATION_1_QUALITATIVE_STRENGTH
        Note that binary proejcts don't prepend the target name to the front:
        This function converts that format to be:
        - Weather_EXPLANATION_1_STRENGTH -> explanation_1_strength
        Parameters
        ----------
        column_label : str
            The column name being converted
        Returns
        -------
        str
            The new column name
        """
        s = re.search(re.escape("EXPLANATION"), column_label)
        if s:
            if s.start() == 0:
                return column_label.lower()
            else:
                return f"{column_label[s.start() : len(column_label)]}".lower()
        else:
            return column_label
