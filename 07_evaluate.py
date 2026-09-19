"""Step 7: تقييم الموديل المحفوظ + رسومات."""
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from pathlib import Path

Path("plots").mkdir(exist_ok=True)

saved = joblib.load("titanic_best_model.pkl")
model = saved["model"]
features = saved["features"]

df = pd.read_csv("titanic_clean.csv")
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_child"] = (df["age"] < 12).astype(int)
X = df[features]
y = df["survived"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# 1. Confusion matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax)
fig.savefig("plots/confusion_matrix.png")
print("saved plots/confusion_matrix.png")

# 2. ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0, 1], [0, 1], linestyle="--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
fig2.savefig("plots/roc_curve.png")
print(f"saved plots/roc_curve.png (AUC={roc_auc:.4f})")

# 3. Survival rate plots (sex / pclass)
fig3, axes = plt.subplots(1, 2, figsize=(10, 4))
df.groupby("sex")["survived"].mean().plot(kind="bar", ax=axes[0], title="Survival by sex")
df.groupby("pclass")["survived"].mean().plot(kind="bar", ax=axes[1], title="Survival by pclass")
fig3.tight_layout()
fig3.savefig("plots/survival_rates.png")
print("saved plots/survival_rates.png")
