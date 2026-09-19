"""Step 1: download the Titanic dataset and save it as CSV."""
from pathlib import Path

import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent

# Load the built-in Titanic dataset (891 passengers)
df = sns.load_dataset("titanic")

# Save it next to this script so we always have a local copy
df.to_csv(BASE_DIR / "titanic.csv", index=False)

print("shape:", df.shape)   # rows x columns
print(df.head(3))           # first 3 rows
