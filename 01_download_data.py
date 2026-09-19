"""Step 1: download the Titanic dataset and save it as CSV."""
import seaborn as sns

# Load the built-in Titanic dataset (891 passengers)
df = sns.load_dataset("titanic")

# Save it next to this script so we always have a local copy
df.to_csv("titanic.csv", index=False)

print("shape:", df.shape)   # rows x columns
print(df.head(3))           # first 3 rows
