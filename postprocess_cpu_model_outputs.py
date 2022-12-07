import json
import boto3
import logging

from botocore.exceptions import ClientError
from ast import literal_eval
from typing import List, Dict
from collections import defaultdict


logging.getLogger().setLevel(logging.INFO)

pillars_1d_tags = [
    "Covid-19",
    "Casualties",
    "Context",
    "Displacement",
    "Humanitarian Access",
    "Shock/Event",
    "Information And Communication",
]

pillars_2d_tags = [
    "At Risk",
    "Priority Interventions",
    "Capacities & Response",
    "Humanitarian Conditions",
    "Impact",
    "Priority Needs",
]

secondary_tags = [
    "age",
    "gender",
    "affected_groups",
    "specific_needs_groups",
    "severity",
]


def get_preds_entry(
    preds_column: Dict[str, float],
    return_at_least_one=True,
    ratio_nb=1,
    return_only_one=False,
):
    if return_only_one:
        preds_entry = [
            sub_tag
            for sub_tag, ratio in preds_column.items()
            if ratio == max(list(preds_column.values()))
        ]
    else:
        preds_entry = [
            sub_tag for sub_tag, ratio in preds_column.items() if ratio > ratio_nb
        ]
        if return_at_least_one and len(preds_entry) == 0:
            preds_entry = [
                sub_tag
                for sub_tag, ratio in preds_column.items()
                if ratio == max(list(preds_column.values()))
            ]
    return preds_entry


def get_predictions_all(
    ratios_entries: List[Dict[str, float]],
    pillars_2d=pillars_2d_tags,
    pillars_1d=pillars_1d_tags,
    ratio_nb: int = 1,
):

    predictions = defaultdict(list)
    for ratio_proba_threshold_one_entry in ratios_entries:
        returns_sectors = ratio_proba_threshold_one_entry["primary_tags"]["sectors"]

        subpillars_2d_tags = ratio_proba_threshold_one_entry["primary_tags"][
            "subpillars_2d"
        ]
        subpillars_1d_tags = ratio_proba_threshold_one_entry["primary_tags"][
            "subpillars_1d"
        ]

        ratios_sectors_subpillars_2d = list(returns_sectors.values()) + list(
            subpillars_2d_tags.values()
        )

        if any([item >= ratio_nb for item in ratios_sectors_subpillars_2d]):
            preds_2d = get_preds_entry(subpillars_2d_tags, True, ratio_nb)
            preds_sectors = get_preds_entry(returns_sectors, True, ratio_nb)

        else:
            preds_2d = []
            preds_sectors = []

        predictions["sectors"].append(preds_sectors)
        predictions["subpillars_2d"].append(preds_2d)

        preds_1d = get_preds_entry(subpillars_1d_tags, False, ratio_nb)
        predictions["subpillars_1d"].append(preds_1d)

        returns_sec_tags = ratio_proba_threshold_one_entry["secondary_tags"]

        for secondary_tag in [
            "age",
            "gender",
            "affected_groups",
            "specific_needs_groups",
        ]:
            preds_one_sec_tag = get_preds_entry(
                returns_sec_tags[secondary_tag], False, ratio_nb
            )

            predictions[secondary_tag].append(preds_one_sec_tag)

        severity_tags = returns_sec_tags["severity"]
        if any(["Humanitarian Conditions" in item for item in preds_2d]):
            preds_severity = get_preds_entry(severity_tags, True, ratio_nb, True)
        else:
            preds_severity = []
        predictions["severity"].append(preds_severity)

    return predictions


def flatten(t: List[List]) -> List:
    return [item for sublist in t for item in sublist]


def convert_current_dict_to_previous_one(
    ratios_one_entry: Dict[str, float]
) -> Dict[str, Dict[str, float]]:

    # prim tags
    primary_tags_results = {"sectors": {}, "subpillars_2d": {}, "subpillars_1d": {}}

    # sec tags
    secondary_tags_results = {
        "age": {},
        "gender": {},
        "affected_groups": {},
        "specific_needs_groups": {},
        "severity": {},
    }

    for tag, number in ratios_one_entry.items():
        tag_levels = tag.split("->")
        if "subpillars" == tag_levels[0]:
            assert tag_levels[1] in pillars_1d_tags or tag_levels[1] in pillars_2d_tags

            if tag_levels[1] in pillars_1d_tags:
                subpillar_name = "subpillars_1d"
            else:
                subpillar_name = "subpillars_2d"

            primary_tags_results[subpillar_name]["->".join(tag_levels[1:])] = number

        elif "secondary_tags" == tag_levels[0]:
            assert tag_levels[1] in secondary_tags

            secondary_tags_results[tag_levels[1]][tag_levels[2]] = number

        else:
            if "sectors" == tag_levels[1]:
                primary_tags_results["sectors"][tag_levels[2]] = number

    outputs = {
        "primary_tags": primary_tags_results,
        "secondary_tags": secondary_tags_results,
    }

    return outputs


def get_endpoint_probas(df, endpoint_name):
    client = boto3.session.Session().client(
        "sagemaker-runtime", region_name="us-east-1"
    )

    all_outputs = []
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(df),
            ContentType="application/json; format=pandas-split",
        )
    except ClientError as error:
        logging.error(f"Error occurred while getting predictions: {error}")
        return []
    all_outputs = response["Body"].read().decode("ascii")

    return all_outputs

def output_all_preds(entry, endpoint_name):
    data = {
        "columns": ["excerpt", "return_type"],
        "index": [0],
        "data": [[entry, "default_analyis"]]
    }

    data["columns"].append("analyis_framework_id")
    data["columns"].append("return_prediction_labels")
    data["columns"].append("interpretability")
    data["columns"].append("ratio_interpreted_labels")
    data["columns"].append("attribution_type")
    data["columns"].append("output_backbone_embeddings")
    data["columns"].append("pooling_type")
    data["columns"].append("finetuned_task")
    data["columns"].append("embeddings_return_type")

    data["data"][0].append("all")
    data["data"][0].append(True)
    data["data"][0].append(False)
    data["data"][0].append(0.5)
    data["data"][0].append("Layer DeepLift")
    data["data"][0].append(False)
    data["data"][0].append("['cls', 'mean_pooling']")
    data["data"][0].append("['first_level_tags', 'secondary_tags', 'subpillars']")
    data["data"][0].append("array")

    endpoint_outputs = get_endpoint_probas(
        df=data,
        endpoint_name=endpoint_name
    )
    if endpoint_outputs:
        eval_batch = literal_eval(endpoint_outputs)
        output_ratios = eval_batch["raw_predictions"]

        thresholds = eval_batch["thresholds"]

        clean_thresholds = convert_current_dict_to_previous_one(thresholds)

        clean_outputs = [
            convert_current_dict_to_previous_one(one_entry_preds)
            for one_entry_preds in output_ratios
        ]

        final_predictions = get_predictions_all(clean_outputs)

        return clean_outputs, clean_thresholds, dict(final_predictions), False
    
    return [], {}, {}, True

