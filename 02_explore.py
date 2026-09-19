"""Step 2: quick exploration of the raw data."""
import pandas as pd

from common import DATA_RAW

df = pd.read_csv(DATA_RAW)
print("shape:", df.shape)
print(df.head())
print(df.dtypes)

print("\n--- target distribution ---")
target = "survived"
print(df[target].value_counts(normalize=True).to_string())
print("survival rate:", round(df[target].mean(), 4))

print("\n--- numeric summary ---")
print(df.describe().to_string())

print("\n--- missing values ---")
print(df.isnull().sum().to_string())

print("\n--- duplicates:", int(df.duplicated().sum()))

# alive is a post-outcome label: verify it mirrors survived (leakage check),
# which is why we must never use it as a feature
if "alive" in df.columns:
    alive_map = {"yes": 1, "no": 0}
    print("alive == survived in all rows:",
          bool((df["alive"].map(alive_map) == df["survived"]).all()))

print("\n--- survival by sex / pclass ---")
print(df.groupby("sex")[target].mean().to_string())
print(df.groupby("pclass")[target].mean().to_string())
