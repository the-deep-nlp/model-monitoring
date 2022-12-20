import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, DataDriftTable

class FeatureDrift:
    """
    Calculates the per project feature drift of excerpts
    Input: DataFrame with columns 'project_id', 'embeddings'
    Output: DataFrame containing columns 'reference_project_id', 'current_project_id',
            'reference_dataset_len', 'current_dataset_len', 'drift_share', 'number_of_columns',
            'number_of_drifted_columns', 'share_of_drifted_columns', 'dataset_drift'
    """
    def __init__(self,
        ref_df: pd.DataFrame,
        cur_df: pd.DataFrame
    ):
        self.reference_df = ref_df
        self.current_df = cur_df
        
        self.data_drift_dataset_report = Report(
            metrics=[DatasetDriftMetric()]
        )
    
    def _process_embeddings(self, embeddings: pd.Series) -> List[float]:
        """
        Reads the embeddings and gets the list
        Input: Embedding Series
        Output: Embedding List
        """
        return embeddings.apply(eval).apply(np.array).to_list()
    
    def _project_id_based_mask(self,
        df: pd.DataFrame,
        project_id: float=None
    ) -> pd.DataFrame:
        """
        Input: DataFrame with column 'project_id'
        Output: Filtered DataFrame if 'project_id' is passed in argument
        """
        if project_id:
            mask = df["project_id"] == project_id
            return df[mask]
    
        return df

    def compute_feature_drift(self, n_samples: int=500, random_state: int=5432) -> pd.DataFrame:
        """
        Computes the feature drift based on feature embeddings
        Input: DataFrame containing columns 'project_id', 'embeddings'
        Output: DataFrame containing columns 'reference_project_id', 'current_project_id',
                'reference_dataset_len', 'current_dataset_len', 'drift_share', 'number_of_columns',
                'number_of_drifted_columns', 'share_of_drifted_columns', 'dataset_drift'
        """
        final_result = list()
        reference_project_ids = list(self.reference_df["project_id"].unique())
        current_project_ids = list(self.current_df["project_id"].unique())

        for project_id in tqdm(reference_project_ids):
            if project_id in current_project_ids:
                reference_df = self._project_id_based_mask(self.reference_df, project_id)
                current_df = self._project_id_based_mask(self.current_df, project_id)
                
                reference_embedding_lst = self._process_embeddings(reference_df["embeddings"])
                current_embedding_lst = self._process_embeddings(current_df["embeddings"])

                reference_df = pd.DataFrame(reference_embedding_lst).sample(n=n_samples, random_state=random_state)
                current_df = pd.DataFrame(current_embedding_lst).sample(n=n_samples, random_state=random_state)

                if len(reference_df) and len(current_df):
                    self.data_drift_dataset_report.run(
                        reference_data=reference_df,
                        current_data=current_df
                    )

                    temp_result = {}
                    temp_result["reference_project_id"] = project_id
                    temp_result["current_project_id"] = project_id
                    temp_result["reference_dataset_len"] = len(reference_df)
                    temp_result["current_dataset_len"] = len(current_df)
                    data_drift_report = self.data_drift_dataset_report.as_dict()
                    # Note the keys might change in the future
                    temp_result["drift_share"] = data_drift_report["metrics"][0]["result"]["drift_share"]
                    temp_result["number_of_columns"] = data_drift_report["metrics"][0]["result"]["number_of_columns"]
                    temp_result["number_of_drifted_columns"] = data_drift_report["metrics"][0]["result"]["number_of_drifted_columns"]
                    temp_result["share_of_drifted_columns"] = temp_result["drift_share"] = data_drift_report["metrics"][0]["result"]["share_of_drifted_columns"]
                    temp_result["dataset_drift"] = temp_result["drift_share"] = data_drift_report["metrics"][0]["result"]["dataset_drift"]
                    temp_result["generated_at"] = datetime.date.today()

                    final_result.append(temp_result)
        
        return pd.DataFrame.from_records(final_result)
    

if __name__ == "__main__":
    reference_data_path = "csvfiles/data_with_embeddings_22000.csv"
    current_data_path = "csvfiles/sampled_data_with_embeddings_testset.csv"
    
    reference_data_df = pd.read_csv(reference_data_path)
    current_data_df = pd.read_csv(current_data_path)

    feature_drift = FeatureDrift(reference_data_df, current_data_df)
    df = feature_drift.compute_feature_drift(n_samples=10)
    print(df)

