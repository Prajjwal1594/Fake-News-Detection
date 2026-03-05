import pandas as pd
import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Load your real dataset ─────────────────────────────────────────────────────
df = pd.read_csv(
    r"C:\Users\p\Desktop\fakeshield_final\fakeshield\data\fake_or_real_news.csv",
    usecols=["title", "text", "label"],
    low_memory=False
)

# Keep only valid labels
df = df[df["label"].isin(["FAKE", "REAL"])]

# Combine title + text for the model input
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["text"] = df["text"].str.strip()

# Convert labels to 0/1
df["label"] = df["label"].map({"REAL": 0, "FAKE": 1})

# Keep all three columns
df = df[["title", "text", "label"]].dropna()

print("Clean shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Label counts:\n", df["label"].value_counts())

df.to_csv(
    r"C:\Users\p\Desktop\fakeshield_final\fakeshield\data\fakenews_clean.csv",
    index=False
)
print("\nSaved as fakenews_clean.csv ✓")