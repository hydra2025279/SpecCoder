import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font to Helvetica
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica"]

# Data points
categories = ["SWE-Agent", "Aider", "Moatless", "Agentless", "SpecRover"]
values = [18, 26.33, 26.67, 27.33, 48.3]

# Create horizontal bar chart with narrower bars and smaller spacing
plt.figure(figsize=(5.2, 2.5))  # Adjust figure height to make bars closer
plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.163)
plt.subplots_adjust(right=0.95)
bars = plt.barh(categories, values, height=0.66)  # Slightly reduce bar height
plt.xlabel("Precision (%)")
plt.title("Comparison of Precision on SWE-Bench Lite")


plt.grid(axis="x", linestyle="--", alpha=0.7)

# Annotate values at the end of the bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_width() + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value}",
        va="center",
        ha="left",
        color="black",
    )

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)

plt.savefig("precision-bar-chart.pdf", format="pdf")
