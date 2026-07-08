import matplotlib

matplotlib.use("Agg")

import json
import os
from os.path import join
from pprint import pprint

import venn

from common import *

# get the data

specrover_resolved = get_resolved("lite/specrover")
agentless_resolved = get_resolved("lite/agentless")
moatless_resolved = get_resolved("lite/moatless")
opendevin_resolved = get_resolved("lite/opendevin")
aider_resolved = get_resolved("lite/aider")

sets = {
    "SpecRover": set(specrover_resolved),
    "Agentless": set(agentless_resolved),
    "Moatless": set(moatless_resolved),
    # "OpenDevin": set(opendevin_resolved),
    "Aider": set(aider_resolved),
}

labels = venn.get_labels(sets.values(), fill=["number"])

# fig, ax = venn.venn(sets)


# colors = ['red', 'blue', 'green', 'purple', 'orange']

fig, ax = venn.venn4(labels, names=sets.keys())

fig.savefig("venn.pdf", format="pdf")

# # Create the Venn diagram
# venn(sets, fmt="{size}")

# # Customize the plot (optional)
# plt.title('Open-source Venn Diagram Example')
# plt.show()


specrover_unique = (
    set(specrover_resolved)
    - set(agentless_resolved)
    - set(moatless_resolved)
    # - set(opendevin_resolved)
    - set(aider_resolved)
)

print("SpecRover unique: ", len(specrover_unique))
pprint(sorted(list(specrover_unique)))
