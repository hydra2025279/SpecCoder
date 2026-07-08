"""
Moatless time is not reported, but can be calculated from the traj files.
"""

import os
import json
from os.path import join as pjoin


traj_dir = "lite/moatless/trajs"

traj_files = os.listdir(traj_dir)

durations = []


for f in traj_files:
    full_f = pjoin(traj_dir, f)
    with open(full_f, "r") as f:
        traj = json.load(f)

    time = traj["info"]["duration"]
    durations.append(time)


# compute average
avg = sum(durations) / len(durations)
print(f"Average moatless time: {avg:.2f} seconds")
