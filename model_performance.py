import datetime
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
#from sklearn.metrics import multilabel_confusion_matrix

from constants import (
    SECTORS,
    SUBPILLARS_1D,
    SUBPILLARS_2D,
    CATEGORIES
)

class ModelPerformance:
    """
    Input: DataFrame that has (atleast) columns 'entry_id', 'excerpt', 'sectors', 'subpillars_1d' 'subpillars_2d',
            'sectors_pred', 'subpillars_1d_pred', 'subpillars_2d_pred'
    Outputs: DataFrame based on the method called.
    """
    def __init__(self, df: pd.DataFrame):
        self.dataframe = df
        self.categories = CATEGORIES

        self._preprocess()
        self._create_mlb()
        self._label_transform()
    
    def validate_fields(self):
        # Check if all the required fields are there in the input df
        pass
    
    def _preprocess(self):
        """
        Preprocess the multi-labels in the dataframe which converts them to be list
        """
        for category in self.categories:
            self.dataframe[category] = self.dataframe[category].apply(literal_eval)
            self.dataframe[f"{category}_pred"] = self.dataframe[f"{category}_pred"].apply(literal_eval)
    
    def _category_to_mlb(self) -> dict:
        # note: update the dict based on the CATEGORIES
        """
        Creates the mappings between Categories and multi-label encoder objects
        """
        return {
            "sectors": self.mlb_sectors,
            "subpillars_1d": self.mlb_subpillars_1d,
            "subpillars_2d": self.mlb_subpillars_2d
        }
    
    def _category_to_tags(self) -> dict:
        return {
            "sectors": SECTORS,
            "subpillars_1d": SUBPILLARS_1D,
            "subpillars_2d": SUBPILLARS_2D
        }
    
    def _create_mlb(self):
        """
        Create objects for the multi-label encoding for all the categories
        """
        self.mlb_sectors = MultiLabelBinarizer()
        self.mlb_sectors.fit_transform([SECTORS])
        
        self.mlb_subpillars_1d = MultiLabelBinarizer()
        self.mlb_subpillars_1d.fit_transform([SUBPILLARS_1D])

        self.mlb_subpillars_2d = MultiLabelBinarizer()
        self.mlb_subpillars_2d.fit_transform([SUBPILLARS_2D])

    def _label_transform(self):
        """
        Transforms the NLP tags to one hot encoding both for ground truth and predicted labels
        Input: DataFrame with columns 'sectors', 'subpillars_1d', 'subpillars_2d', 'sectors_pred',
                'subpillars_1d_pred', 'subpillars_2d_pred'
        Output: DataFrame with columns 'sectors_transformed', 'sectors_pred_transformed',
                'subpillars_1d_transformed', 'subpillars_1d_pred_transformed',
                'subpillars_2d_transformed', 'subpillars_2d_pred_transformed'
        """
        cat_to_mlb = self._category_to_mlb()
        for category in self.categories:
            self.dataframe[f"{category}_transformed"] = list(
                cat_to_mlb[category].transform(
                    list(
                        self.dataframe[category]
                    )
                )
            )
            self.dataframe[f"{category}_pred_transformed"] = list(
                cat_to_mlb[category].transform(
                    list(
                        self.dataframe[f"{category}_pred"]
                    )
                )
            )

    def project_wise_perf_metrics(self, metrics_average_type: str="macro") -> pd.DataFrame:
        """
        Calculates project wise performance metrics(average) on all categories
        Input: DataFrame with columns 'sectors_transformed', 'sectors_pred_transformed',
                'subpillars_1d_transformed', 'subpillars_1d_pred_transformed',
                'subpillars_2d_transformed', 'subpillars_2d_pred_transformed
        Output: DataFrame with rows: project_ids, and columns like
                'sectors_precision', 'sectors_recall', 'sectors_f1score', 'sectors_support',
                'subpillars_1d_precision', 'subpillars_1d_recall', 'subpillars_1d_f1score', 'subpillars_1d_support',
                'subpillars_2d_precision', 'subpillars_2d_recall', 'subpillars_2d_f1score', 'subpillars_2d_support'
        """
        project_perf_metrics = {}

        for category in self.categories:
            for project_grp_gt, project_grp_pred in zip(
                list(self.dataframe.groupby("project_id")[f"{category}_transformed"]),
                list(self.dataframe.groupby("project_id")[f"{category}_pred_transformed"])
            ):
                project_id = project_grp_gt[0]
                precision, recall, f1score, support = precision_recall_fscore_support(
                    list(project_grp_gt[1]),
                    list(project_grp_pred[1]),
                    average=metrics_average_type,
                    zero_division=0
                )
                if project_id not in project_perf_metrics:
                    project_perf_metrics[project_id] = {
                        f"{category}_precision": precision,
                        f"{category}_recall": recall,
                        f"{category}_f1score": f1score,
                        f"{category}_support": support
                    }
                project_perf_metrics[project_id].update({
                    f"{category}_precision": precision,
                    f"{category}_recall": recall,
                    f"{category}_f1score": f1score,
                    f"{category}_support": support
                })
        final_df = pd.DataFrame(project_perf_metrics).T # (x, y) = (project_id, perf_metrics)
        final_df.reset_index(inplace=True)
        final_df = final_df.rename(columns={'index': 'project_id'})
        final_df["generated_at"] = datetime.date.today()
        return final_df
    
    def per_tag_perf_metrics(self) -> pd.DataFrame:
        """
        Performance Metrics of tags in overall projects
        Input: DataFrame with columns like 'sectors_transformed', 'sectors_pred_transformed',
                'subpillars_1d_transformed', 'subpillars_1d_pred_transformed',
                'subpillars_2d_transformed', 'subpillars_2d_pred_transformed'
        Output: DataFrame with rows that contains all the tags from all the cateogories
                and columns like 'Precision', 'Recall', 'F1Score'
        """
        cat_to_tags = self._category_to_tags()
        tag_precision_perf_metrics = {}
        tag_recall_perf_metrics = {}
        tag_f1score_perf_metrics = {}
        tag_support_perf_metrics = {}

        for category in self.categories:
            precision, recall, f1score, support = precision_recall_fscore_support(
                list(self.dataframe[f"{category}_transformed"]),
                list(self.dataframe[f"{category}_pred_transformed"]),
                zero_division=0
            )
            for tag, metric_val in zip(cat_to_tags[category], precision):
                tag_precision_perf_metrics[tag] = metric_val
            for tag, metric_val in zip(cat_to_tags[category], recall):
                tag_recall_perf_metrics[tag] = metric_val
            for tag, metric_val in zip(cat_to_tags[category], f1score):
                tag_f1score_perf_metrics[tag] = metric_val
            for tag, metric_val in zip(cat_to_tags[category], support):
                tag_support_perf_metrics[tag] = metric_val

        per_tag_precision_df = pd.DataFrame.from_dict(tag_precision_perf_metrics, orient="index", columns=["precision"])
        per_tag_recall_df = pd.DataFrame.from_dict(tag_recall_perf_metrics, orient="index", columns=["recall"])
        per_tag_f1score_df = pd.DataFrame.from_dict(tag_f1score_perf_metrics, orient="index", columns=["f1score"])
        per_tag_support_df = pd.DataFrame.from_dict(tag_support_perf_metrics, orient="index", columns=["support"])

        final_df = pd.concat([
                per_tag_precision_df,
                per_tag_recall_df,
                per_tag_f1score_df,
                per_tag_support_df
            ], axis=1)
        final_df.reset_index(inplace=True)
        final_df = final_df.rename(columns={'index': 'tags'})
        final_df["generated_at"] = datetime.date.today()
        return final_df

    
    def all_projects_perf_metrics(self, metrics_average_type: str="macro"):
        """
        Performance Metrics of the all the projects altogether
        Input: A DataFrame with columns 'sectors_transformed', 'sectors_pred_transformed',
                'subpillars_1d_transformed', 'subpillars_1d_pred_transformed',
                'subpillars_2d_transformed', 'subpillars_2d_pred_transformed'
        Output: A DataFrame with indices 'sectors', 'subpillars_1d', 'subpillars_2d', and
                columns 'precision', 'recall', 'f1score', 'support'
        """
        all_projects_performance_metrics = {}

        for category in self.categories:
            precision, recall, f1score, support = precision_recall_fscore_support(
                list(self.dataframe[f"{category}_transformed"]),
                list(self.dataframe[f"{category}_pred_transformed"]),
                average=metrics_average_type,
                zero_division=0
            )
            
            all_projects_performance_metrics[category] = {
                "precision": precision,
                "recall": recall,
                "f1score": f1score,
                "support": support
            }
        final_df = pd.DataFrame(all_projects_performance_metrics).T # (x, y) = (project_id, perf_metrics)
        final_df.reset_index(inplace=True)
        final_df = final_df.rename(columns={'index': 'categories'})
        final_df["generated_at"] = datetime.date.today()
        return final_df
    

    def completely_matched_tags(self, lst: list) -> int:
        return (lst[0] == lst[1]).sum()

    def missing_tags(self, lst: list) -> int:
        return (lst[0] > lst[1]).sum()
    
    def wrong_tags(self, lst: list) -> int:
        return (lst[0]<lst[1]).sum()
    
    def calculate_ratios(self) -> pd.DataFrame:
        """
        Calculates the ratios of the categories
        Input: DataFrame with columns 'sectors_transformed', 'sectors_pred_transformed',
                'subpillars_1d_transformed', 'subpillars_1d_pred_transformed',
                'subillars_2d_transformed', 'subillars_2d_pred_transformed'
        Output: DataFrame with columns 'entry_id', 'project_id',
                'sectors_completely_matched', 'sectors_missing', 'sectors_wrong',
                'subpillars_1d_completely_matched', 'subillars_1d_missing', 'subpillars_1d_wrong',
                'subpillars_2d_completely_matched, 'subpillars_2d_missing', 'subpillars_2d_wrong'
        """
        cat_to_mlb = self._category_to_mlb()
        ratios_df = pd.DataFrame()

        for category in self.categories:
            ratios_df[f"{category}_completely_matched"] = pd.Series(list(
                map(
                    self.completely_matched_tags,
                    list(zip(np.array(self.dataframe[f"{category}_transformed"]), np.array(self.dataframe[f"{category}_pred_transformed"])))
                )
            ))
            ratios_df[f"{category}_completely_matched"] /= len(cat_to_mlb[category].classes_)

            ratios_df[f"{category}_missing"] = pd.Series(list(
                map(self.missing_tags,
                list(zip(np.array(self.dataframe[f"{category}_transformed"]), np.array(self.dataframe[f"{category}_pred_transformed"])))
                )
            ))
            ratios_df[f"{category}_missing"] /= len(cat_to_mlb[category].classes_)
            
            ratios_df[f"{category}_wrong"] = pd.Series(list(
                map(self.wrong_tags,
                list(zip(np.array(self.dataframe[f"{category}_transformed"]), np.array(self.dataframe[f"{category}_pred_transformed"])))
                )
            ))
            ratios_df[f"{category}_wrong"] /= len(cat_to_mlb[category].classes_)

        # Adds entry_id and project_id columns to the df
        ratios_df = pd.concat([
            ratios_df,
            self.dataframe[["entry_id", "project_id"]]
        ], axis=1)

        ratios_df["generated_at"] = datetime.date.today()
        return ratios_df
    
    def per_project_calc_ratios(self) -> pd.DataFrame:
        """
        Input: DataFrame with columns 'entry_id', 'excerpt', 'sectors', 'subpillars_1d', 'subpillars_2d'
        Output: DataFrame with columns 'sectors_completely_matched_mean', 'sectors_missing_mean', 'sectors_wrong_mean',
                'subpillars_1d_completely_matched_mean', 'subpillars_1d_missing_mean', 'subpillars_1d_wrong_mean',
                'subpillars_2d_completely_matched_mean', 'subpillars_2d_missing_mean', 'subpillars_2d_wrong_mean'
        """
        ratios_df = self.calculate_ratios()
        final_df = pd.DataFrame()
        for category in self.categories:
            final_df[f"{category}_completely_matched_mean"] = ratios_df.groupby(["project_id"])[f"{category}_completely_matched"].mean()
            final_df[f"{category}_missing_mean"] = ratios_df.groupby(["project_id"])[f"{category}_missing"].mean()
            final_df[f"{category}_wrong_mean"] = ratios_df.groupby(["project_id"])[f"{category}_wrong"].mean()
        
        final_df.reset_index(inplace=True)
        final_df = final_df.rename(columns={'index': 'project_id'})
        final_df["generated_at"] = datetime.date.today()
        return final_df

if __name__ == "__main__":
    df = pd.read_csv("csvfiles/sampled_data_with_predictions_testset.csv").sample(n=50, random_state=1234).reset_index()
    modelperf = ModelPerformance(df)
    
    df1 = modelperf.project_wise_perf_metrics()
    #print(df1)
    df2 = modelperf.per_tag_perf_metrics()
    #print(df2)
    df3 = modelperf.all_projects_perf_metrics()
    #print(df3)
    df4 = modelperf.calculate_ratios()
    #print(df4)
    df5 = modelperf.per_project_calc_ratios()
    #print(df5)



