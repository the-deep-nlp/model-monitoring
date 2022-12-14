import json
import logging
import boto3
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from botocore.exceptions import ClientError

ENDPOINT_NAME = "main-model-cpu"
AWS_REGION = "us-east-1"
BATCH = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embeddings:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.sg_client = boto3.session.Session().client(
            "sagemaker-runtime", region_name=AWS_REGION
        )

    def _create_df(self, entries: str) -> dict:
        df = pd.DataFrame({"excerpt": entries, "index": range(len(entries))})
        df["return_type"] = "default_analyis"
        df["analyis_framework_id"] = "all"

        df["interpretability"] = False
        df["ratio_interpreted_labels"] = 0.5
        df["return_prediction_labels"] = False

        df["output_backbone_embeddings"] = True
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
            embeddings = json.loads(response["Body"].read().decode("ascii"))
            return embeddings["output_backbone"]
        except ClientError as err:
            logger.error(str(err))
            return []

    def _get_embeddings(self, excerpt: str) -> List:
        df_json = self._create_df(excerpt)
        embeddings = self.invoke_endpoint(df_json)
        return embeddings
    
    def generate_embeddings(self) -> pd.DataFrame:
        temp_series =  pd.Series([], dtype=pd.StringDtype())
        for entries_batch in tqdm(
            np.array_split(self.dataframe, BATCH)
        ):
            temp_series = pd.concat(
                [
                    temp_series,
                    pd.Series(self._get_embeddings(entries_batch["excerpt"]))
                ], axis=0, ignore_index=True
            )
        final_df = pd.concat([
            self.dataframe["entry_id"], temp_series.to_frame("embeddings")],
            axis=1
        )
        return final_df


if __name__ == "__main__":
    dataset_path = "csvfiles/test_v0.7.1.csv"
    df = pd.read_csv(dataset_path).sample(50).reset_index()

    embeddings = Embeddings(df)
    df = embeddings.generate_embeddings()

    df.to_csv("results_with_embeddings.csv")



