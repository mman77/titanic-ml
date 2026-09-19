"""Step 7: evaluate the saved model + plots + error analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from common import DATA_CLEAN, FEATURES, PLOTS_DIR, add_features, load_model, split

PLOTS_DIR.mkdir(exist_ok=True)

model = load_model()["model"]

df = add_features(pd.read_csv(DATA_CLEAN))
X = df[FEATURES]
y = df["survived"]
_, X_test, _, y_test = split(X, y)  # same seed as training -> same holdout

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# 1. confusion matrix with readable labels
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm, display_labels=["died", "survived"]).plot(ax=ax)
ax.set_title("Confusion matrix (holdout)")
fig.savefig(PLOTS_DIR / "confusion_matrix.png")
plt.close(fig)
print("saved plots/confusion_matrix.png")

tn, fp, fn_count, tp = cm.ravel()
print(f"TP={tp} FN={fn_count} FP={fp} TN={tn} | "
      f"recall(survived)={tp / (tp + fn_count):.3f} | "
      f"precision(survived)={tp / (tp + fp):.3f}")

# 2. ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0, 1], [0, 1], linestyle="--", label="chance")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC curve (holdout)")
ax2.legend()
fig2.savefig(PLOTS_DIR / "roc_curve.png")
plt.close(fig2)
print(f"saved plots/roc_curve.png (AUC={roc_auc:.4f})")

# 3. error analysis: where does the model fail? (always show group size n)
err = X_test.copy()
err["y_true"] = y_test.values
err["y_pred"] = pred
err["correct"] = err["y_true"] == err["y_pred"]
print("\n--- accuracy by sex (mean, n) ---")
print(err.groupby("sex")["correct"].agg(["mean", "count"]).round(4).to_string())
print("\n--- accuracy by pclass (mean, n) ---")
print(err.groupby("pclass")["correct"].agg(["mean", "count"]).round(4).to_string())
print("\n--- false negatives sample (died predicted, actually survived) ---")
fn = err[(err["y_true"] == 1) & (err["y_pred"] == 0)]
print(fn.head(5).to_string())
print(f"false negatives: {len(fn)} / false positives: {fp}")
