import json
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix

from constants import (
    SECTORS,
    SUBPILLARS_1D,
    SUBPILLARS_2D,
    CATEGORIES
)

class ModelPerformance:
    def __init__(self, df):
        self.dataframe = df
        self.categories = CATEGORIES
    
    def validate_fields(self):
        # Check if all the required fields are there in the input df
        pass
    
    def _preprocess(self):
        for category in self.categories:
            self.dataframe[category] = self.dataframe[category].apply(literal_eval)
            self.dataframe[f"{category}_pred"] = self.dataframe[f"{category}_pred"].apply(literal_eval)
    
    def _category_to_mlb(self):
        # note: update the dict based on the CATEGORIES
        return {
            "sectors": self.mlb_sectors,
            "subpillars_1d": self.mlb_subpillars_1d,
            "subpillars_2d": self.mlb_subpillars_2d
        }
    
    def _category_to_tags(self):
        return {
            "sectors": SECTORS,
            "subpillars_1d": SUBPILLARS_1D,
            "subpillars_2d": SUBPILLARS_2D
        }
    
    def _create_mlb(self):
        self.mlb_sectors = MultiLabelBinarizer()
        self.mlb_sectors.fit_transform([SECTORS])
        
        self.mlb_subpillars_1d = MultiLabelBinarizer()
        self.mlb_subpillars_1d.fit_transform([SUBPILLARS_1D])

        self.mlb_subpillars_2d = MultiLabelBinarizer()
        self.mlb_subpillars_2d.fit_transform([SUBPILLARS_2D])

    def _label_transform(self):
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

    def project_wise_perf_metrics(self, metrics_average_type="macro"):
        """
        Calculates project wise performance metrics(average) on all categories
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
        return pd.DataFrame(project_perf_metrics).T # (x, y) = (project_id, perf_metrics)
    
    def per_tag_perf_metrics(self):
        """
        Performance Metrics of tags in overall projects
        """
        cat_to_tags = self._category_to_tags()
        tag_precision_perf_metrics = {}
        tag_recall_perf_metrics = {}
        tag_f1score_perf_metrics = {}

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

        per_tag_precision_df = pd.DataFrame.from_dict(tag_precision_perf_metrics, orient="index", columns=["Precision"])
        per_tag_recall_df = pd.DataFrame.from_dict(tag_recall_perf_metrics, orient="index", columns=["Recall"])
        per_tag_f1score_df = pd.DataFrame.from_dict(tag_f1score_perf_metrics, orient="index", columns=["F1Score"])

        final_df = pd.concat([
                per_tag_precision_df,
                per_tag_recall_df,
                per_tag_f1score_df
            ], axis=1)
        
        return final_df
    
    def all_projects_perf_metrics(self, metrics_average_type="macro"):
        """
        Performance Metrics of the all the projects altogether
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
        return pd.DataFrame(all_projects_performance_metrics).T # (x, y) = (project_id, perf_metrics)
    

    def completely_matched_tags(self, lst):
        return (lst[0] == lst[1]).sum()

    def missing_tags(self, lst):
        return (lst[0] > lst[1]).sum()
    
    def wrong_tags(self, lst):
        return (lst[0]<lst[1]).sum()
    
    def calculate_ratios(self):
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

        # Add entry_id and project_id columns to the df
        ratios_df = pd.concat([
            ratios_df,
            self.dataframe[["entry_id", "project_id"]]
        ], axis=1)
    
        return ratios_df
    
    def per_project_calc_ratios(self):
        ratios_df = self.calculate_ratios()
        final_df = pd.DataFrame()
        for category in self.categories:
            final_df[f"{category}_completely_matched_mean"] = ratios_df.groupby(["project_id"])[f"{category}_completely_matched"].mean()
            final_df[f"{category}_missing_mean"] = ratios_df.groupby(["project_id"])[f"{category}_missing"].mean()
            final_df[f"{category}_wrong_mean"] = ratios_df.groupby(["project_id"])[f"{category}_wrong"].mean()
        
        return final_df

if __name__ == "__main__":
    df = pd.read_csv("csvfiles/sampled_data_with_predictions_testset.csv").sample(n=50, random_state=1234).reset_index()
    modelperf = ModelPerformance(df)
    modelperf._preprocess()
    modelperf._create_mlb()
    modelperf._label_transform()
    #df2 = modelperf.project_wise_perf_metrics()
    modelperf.per_tag_perf_metrics()
    df = modelperf.all_projects_perf_metrics()
    modelperf.calculate_ratios()
    modelperf.per_project_calc_ratios()

    #print(df2)



