"""Step 3: structural cleaning only (no value imputation here).

Missing-value imputation lives inside the model Pipelines (SimpleImputer)
so medians/modes are learned from train folds only. This file only drops
columns that are unusable, redundant, or leaky:
- deck: mostly missing (percentage computed below, not hardcoded)
- alive/class: post-outcome or duplicate labels (alive mirrors survived)
- embark_town/who/adult_male: duplicates of embarked/sex/age
"""
import pandas as pd

from common import DATA_CLEAN, DATA_RAW

df = pd.read_csv(DATA_RAW)
print("before cleaning:")
print(df.isnull().sum().to_string())

deck_missing = df["deck"].isnull().mean()
print(f"\ndeck missing: {deck_missing:.1%} -> dropping column")

df = df.drop(columns=["deck", "alive", "class",
                      "embark_town", "who", "adult_male"])

print("\nafter structural cleaning (remaining NaN is intentional):")
print(df.isnull().sum().to_string())

df.to_csv(DATA_CLEAN, index=False)
print("\nsaved to titanic_clean.csv shape:", df.shape)
