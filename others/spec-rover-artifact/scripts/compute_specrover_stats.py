"""
SpecRover has individual stats. Compute the overall stats and store it.
"""

import json
import os
from datetime import datetime
from os.path import join as pjoin

from common import *

specrover_dir = "lite/specrover/"

task_dirs = []

# collect all task dirs
for d in os.listdir(pjoin(specrover_dir, "applicable_patch")):
    task_dirs.append(pjoin(specrover_dir, "applicable_patch", d))

for d in os.listdir(pjoin(specrover_dir, "matched_but_empty_diff")):
    task_dirs.append(pjoin(specrover_dir, "matched_but_empty_diff", d))

for d in os.listdir(pjoin(specrover_dir, "raw_patch_but_unmatched")):
    task_dirs.append(pjoin(specrover_dir, "raw_patch_but_unmatched", d))


# collect all stats


def parse_time_from_line(line: str):

    date_string = line.split("|")[0].strip()

    # Define the format of the date string
    format_string = "%Y-%m-%d %H:%M:%S.%f"

    # Parse the string into a datetime object
    parsed_datetime = datetime.strptime(date_string, format_string)

    return parsed_datetime


def parse_log_for_time_cost(d):
    """
    Tasks with no patch will not have a cost.json file.
    Parse log file to get those info.
    """
    log_file = pjoin(d, "info.log")
    with open(log_file, "r") as f:
        lines = f.read().splitlines()

    # get the time
    start_line = lines[0]
    end_line = lines[-1]

    start_time = parse_time_from_line(start_line)
    end_time = parse_time_from_line(end_line)

    time_diff = end_time - start_time
    time_diff_s = time_diff.total_seconds()

    # get the cost

    cost = 0
    for line in lines:
        if not "API request cost info:" in line:
            continue
        req_cost = float(line.split("=")[-1])
        cost += req_cost

    return time_diff_s, cost


def collect_stats_from_one_dir(d):
    # get task id
    relative_d = d.split("/")[-1]
    task_id = relative_d.rsplit("_", 2)[0]

    stats_file = pjoin(d, "cost.json")
    if not os.path.exists(stats_file):
        elapsed_seconds, total_cost = parse_log_for_time_cost(d)
        return task_id, elapsed_seconds, total_cost

    with open(stats_file, "r") as f:
        stats = json.load(f)

    return task_id, stats["elapsed_seconds"], stats["total_cost"]


task_seconds = []
task_cost = []

for d in task_dirs:
    task_id, seconds, cost = collect_stats_from_one_dir(d)
    task_seconds.append((task_id, seconds))
    task_cost.append((task_id, cost))


# compute average

def compute_average(id_with_values):
    total = 0
    for _, value in id_with_values:
        total += value
    return total / len(id_with_values)


avg_seconds = compute_average(task_seconds)
avg_cost = compute_average(task_cost)

print(f"Average time: {avg_seconds:.2f} seconds")
print(f"Average cost: {avg_cost:.2f}")


# compute average for resolved tasks

resolvd = get_resolved(specrover_dir)

resolved_task_seconds = [x for x in task_seconds if x[0] in resolvd]
resolved_task_cost = [x for x in task_cost if x[0] in resolvd]

avg_resolved_seconds = compute_average(resolved_task_seconds)
avg_resolved_cost = compute_average(resolved_task_cost)

print(f"Average time for resolved tasks: {avg_resolved_seconds:.2f} seconds")
print(f"Average cost for resolved tasks: {avg_resolved_cost:.2f}")
