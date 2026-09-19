"""Step 4: analysis with counts, two-way pivot, and a saved plot.

Note: age still has NaN here (imputation happens in the model Pipeline),
so age stats below cover the 714 passengers with known age only.
Child definition is unified with the model: age < 12.
"""
import pandas as pd

from common import DATA_CLEAN, PLOTS_DIR

df = pd.read_csv(DATA_CLEAN)

# 1. survival rate + group size (rates without counts mislead on small groups)
print("--- survival rate + count by who-like groups ---")
print(df.groupby("sex")["survived"].agg(["mean", "count"]).to_string())
print(df.groupby("pclass")["survived"].agg(["mean", "count"]).to_string())

# 2. two-way pivot: alone is confounded with sex/pclass, so cross them
print("\n--- survival pivot: sex x pclass ---")
print(pd.pivot_table(df, values="survived", index="sex",
                     columns="pclass", aggfunc="mean").to_string())
print("\n--- survival pivot: alone x sex ---")
print(pd.pivot_table(df, values="survived", index="alone",
                     columns="sex", aggfunc="mean").to_string())

# 3. age on original (non-imputed) values
df["is_child"] = (df["age"] < 12).astype(int)
print("\n--- age known:", int(df['age'].notna().sum()), "/ missing:",
      int(df['age'].isna().sum()))
print(df.groupby("is_child")["survived"].agg(["mean", "count"]).to_string())

# 4. fare: describe + note the fare=0 edge cases
print("\n--- fare describe by survived ---")
print(df.groupby("survived")["fare"].describe().to_string())
print("fare == 0 count:", int((df["fare"] == 0).sum()))

# 5. bar plot (moved here from 07: this is EDA, not model evaluation)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.groupby("sex")["survived"].mean().plot(
    kind="bar", ax=axes[0], title="Survival by sex")
axes[0].set_ylabel("survival rate")
df.groupby("pclass")["survived"].mean().plot(
    kind="bar", ax=axes[1], title="Survival by pclass")
axes[1].set_ylabel("survival rate")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "survival_rates.png")
plt.close(fig)
print("\nsaved plots/survival_rates.png")
