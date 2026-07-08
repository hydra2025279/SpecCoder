import os
import json
from os.path import join


def get_resolved(dir_name):
    report_file = join(dir_name, "report", "report.json")

    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            data = json.load(f)
        return data["resolved"]

    result_file = join(dir_name, "results", "results.json")
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            data = json.load(f)
        return data["resolved"]

    alternative_file = join(dir_name, "results.json")
    if os.path.exists(alternative_file):
        with open(alternative_file, "r") as f:
            data = json.load(f)
        return data["resolved_ids"]

    return []
