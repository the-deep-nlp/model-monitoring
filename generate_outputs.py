import json
import logging
import datetime
import boto3
import pandas as pd
import numpy as np

from typing import List
from tqdm import tqdm
from botocore.exceptions import ClientError

from postprocess_cpu_model_outputs import convert_current_dict_to_previous_one, get_predictions_all

ENDPOINT_NAME = "main-model-cpu"
AWS_REGION = "us-east-1"
BATCH = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationModelOutput:
    """
    Input: DataFrame with fields ['entry_id', 'excerpts'] (at least)
    Output: DataFrame with fields ['entry_id', 'embeddings', 'sectors_pred', 'subpillars_2d_pred',
       'subpillars_1d_pred', 'age_pred', 'gender_pred', 'affected_groups_pred',
       'specific_needs_groups_pred', 'severity_pred']
    """
    def __init__(self,
        dataframe: pd.DataFrame,
        prediction_required: bool=True,
        embeddings_required: bool=True
    ):
        self.dataframe = dataframe
        self.batch = len(self.dataframe)//BATCH
        self.sg_client = boto3.session.Session().client(
            "sagemaker-runtime", region_name=AWS_REGION
        )
        self.prediction_required = prediction_required
        self.embeddings_required = embeddings_required
        self.embeddings = []
        self.predictions = []
        self.thresholds = {}

    def _create_df(self,
        entries: str
    ) -> dict:
        df = pd.DataFrame({"excerpt": entries, "index": range(len(entries))})
        df["return_type"] = "default_analyis"
        df["analyis_framework_id"] = "all"

        df["interpretability"] = False
        df["ratio_interpreted_labels"] = 0.5
        df["return_prediction_labels"] = self.prediction_required

        df["output_backbone_embeddings"] = self.embeddings_required
        df["pooling_type"] = "['mean_pooling']"
        df["finetuned_task"] = "['first_level_tags', 'subpillars']"
        df["embeddings_return_type"] = "list"

        return df.to_json(orient="split")

    def invoke_endpoint(self, backbone_inputs_json: dict) -> List:
        try:
            response = self.sg_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                Body=backbone_inputs_json,
                ContentType="application/json; format=pandas-split"
            )
            return json.loads(response["Body"].read().decode("ascii"))
        except ClientError as err:
            raise Exception(err)

    def _get_outputs(self, excerpt: str) -> None:
        df_json = self._create_df(excerpt)
        outputs = self.invoke_endpoint(df_json)
        if self.embeddings_required and "output_backbone" in outputs:
            self.embeddings = outputs["output_backbone"]
        if self.prediction_required and (
            "raw_predictions" in outputs and
            "thresholds" in outputs):
            self.predictions = outputs["raw_predictions"]
            self.thresholds = outputs["thresholds"]
    
    def column_mapping(self, cols) -> dict:
        return {col: f"{col}_pred" for col in cols}

    def generate_embeddings(self) -> pd.DataFrame:
        return self.embeddings
            
    def generate_predictions(self) -> List[dict]:
        output_ratios = self.predictions

        thresholds = self.thresholds

        clean_thresholds = convert_current_dict_to_previous_one(thresholds)

        clean_outputs = [
            convert_current_dict_to_previous_one(one_entry_preds)
            for one_entry_preds in output_ratios
        ]

        return get_predictions_all(clean_outputs)
    
    def generate_outputs(self) -> pd.DataFrame:
        """
        Returns a dataframe with fields ['entry_id', 'embeddings', 'sectors_pred', 'subpillars_2d_pred',
       'subpillars_1d_pred', 'age_pred', 'gender_pred', 'affected_groups_pred',
       'specific_needs_groups_pred', 'severity_pred']
        """
        embedding_series = pd.Series([], dtype=pd.StringDtype())
        prediction_df = pd.DataFrame([], dtype=pd.StringDtype())
        final_df = pd.DataFrame([])

        for entries_batch in tqdm(
            np.array_split(self.dataframe, self.batch)
        ):
            self._get_outputs(entries_batch["excerpt"])
            if self.embeddings_required:
                embedding_series = pd.concat(
                    [
                        embedding_series,
                        pd.Series(self.embeddings)
                    ], axis=0, ignore_index=True
                )
            if self.prediction_required:
                prediction_df = pd.concat(
                    [
                        prediction_df,
                        pd.DataFrame(self.generate_predictions())
                    ], ignore_index=True
                )
        
        prediction_df.rename(
            columns=self.column_mapping(prediction_df.columns),
            inplace=True
        )

        if self.embeddings_required and self.prediction_required:
            final_df = pd.concat([
                self.dataframe["entry_id"],
                embedding_series.to_frame("embeddings"),
                prediction_df
            ], axis=1)
        elif self.embeddings_required:
            final_df = pd.concat([
                self.dataframe["entry_id"],
                embedding_series.to_frame("embeddings")
            ], axis=1)
        elif self.prediction_required:
            final_df = pd.concat([
                self.dataframe["entry_id"],
                prediction_df
            ], axis=1)
        
        final_df["generated_at"] = datetime.date.today()
        return final_df


if __name__ == "__main__":
    dataset_path = "csvfiles/test_v0.7.1.csv"
    df = pd.read_csv(dataset_path).sample(n=10, random_state=1234).reset_index()

    embeddings = ClassificationModelOutput(
        df,
        prediction_required=True,
        embeddings_required=True
    )
    df = embeddings.generate_outputs()
    #print(df)



