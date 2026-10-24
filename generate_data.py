"""Generates a realistic synthetic campus placement dataset (1500 students).
To use a real Kaggle dataset instead, save it as data/placement_data.csv with the
same column names (see README)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

branch = rng.choice(["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"], N,
                    p=[0.28, 0.18, 0.22, 0.12, 0.12, 0.08])
cgpa = np.clip(rng.normal(7.4, 1.0, N), 5.0, 10.0).round(2)
internships = rng.choice([0, 1, 2, 3], N, p=[0.35, 0.37, 0.20, 0.08])
projects = np.clip(rng.poisson(2.2, N), 0, 6)
certifications = np.clip(rng.poisson(1.8, N), 0, 7)
aptitude = np.clip(rng.normal(60 + (cgpa - 7.4) * 6, 14, N), 15, 100).round(1)
coding = np.clip(rng.normal(55 + (cgpa - 7.4) * 5 + projects * 2, 15, N), 10, 100).round(1)
communication = np.clip(rng.normal(6.2, 1.5, N), 1, 10).round(1)
backlogs = rng.choice([0, 1, 2, 3], N, p=[0.78, 0.13, 0.06, 0.03])

branch_bonus = pd.Series(branch).map({"CSE": 0.6, "IT": 0.6, "ECE": 0.2, "EEE": 0.0,
                                      "Mechanical": -0.2, "Civil": -0.3}).values
score = (-11.6 + 0.9 * cgpa + 0.55 * internships + 0.18 * projects + 0.12 * certifications
         + 0.025 * aptitude + 0.03 * coding + 0.18 * communication
         - 0.9 * backlogs + branch_bonus + rng.normal(0, 0.9, N))
placed = (rng.random(N) < 1 / (1 + np.exp(-score))).astype(int)

df = pd.DataFrame({
    "branch": branch, "cgpa": cgpa, "internships": internships, "projects": projects,
    "certifications": certifications, "aptitude_score": aptitude, "coding_score": coding,
    "communication_score": communication, "backlogs": backlogs, "placed": placed,
})
# introduce a few missing values so the cleaning step is real
for col in ["aptitude_score", "coding_score", "communication_score"]:
    df.loc[rng.choice(N, 25, replace=False), col] = np.nan
df.to_csv("data/placement_data.csv", index=False)
print(f"Saved data/placement_data.csv  shape={df.shape}  placed rate={df.placed.mean():.2%}")
