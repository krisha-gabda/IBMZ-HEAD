
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/neet_dataset.csv")

# Create output folder
output_dir = Path("charts")
output_dir.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------
# 1. NEET vs Non-NEET
# ------------------------------------
counts = df["neet"].map({
    0: "Non-NEET",
    1: "NEET"
}).value_counts().reindex(["Non-NEET", "NEET"])

ax = counts.plot(kind="bar", rot=0)
ax.bar_label(ax.containers[0])

plt.title("NEET vs Non-NEET Learners")
plt.xlabel("Learner Status")
plt.ylabel("Number of Learners")
plt.tight_layout()
plt.savefig(output_dir / "neet_vs_non_neet.png", dpi=200)
plt.show()
plt.close()

# ------------------------------------
# 2. NEET Rate by Qualification Level
# ------------------------------------
qualification_rate = (
    df.groupby("qualification_level")["neet"].mean() * 100
).sort_values(ascending=False)

ax = qualification_rate.plot(kind="bar", rot=35)
ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)

plt.title("NEET Rate by Qualification Level")
plt.xlabel("Qualification Level")
plt.ylabel("NEET Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(output_dir / "neet_by_qualification.png", dpi=200)
plt.show()
plt.close()

# ------------------------------------
# 3. Attendance: NEET vs Non-NEET
# ------------------------------------
df["neet_status"] = df["neet"].map({
    0: "Non-NEET",
    1: "NEET"
})

sns.boxplot(
    data=df,
    x="neet_status",
    y="attendance_pct",
    order=["Non-NEET", "NEET"]
)

plt.title("Attendance: NEET vs Non-NEET")
plt.xlabel("Learner Status")
plt.ylabel("Attendance (%)")
plt.tight_layout()
plt.savefig(output_dir / "attendance_comparison.png", dpi=200)
plt.show()
plt.close()

print("All three graphs have been saved in the charts/ folder.")
