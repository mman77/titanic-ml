"""Step 9: features جديدة + مقارنة متقدمة + metrics.txt
ملحوظة: داتا seaborn مفيهاش Name/Ticket فـ Title مش متاح،
بنستخدم المتاح: who + fare_per_person + fare_log + age_x_pclass.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("titanic_clean.csv")

# --- features جديدة ---
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_child"] = (df["age"] < 12).astype(int)
df["fare_per_person"] = df["fare"] / df["family_size"]
df["fare_log"] = np.log1p(df["fare"])
df["age_x_pclass"] = df["age"] * df["pclass"]

features = ["pclass", "sex", "age", "sibsp", "parch", "fare",
            "embarked", "alone", "family_size", "is_child",
            "fare_per_person", "fare_log", "age_x_pclass",
            "who", "adult_male"]
X = df[features]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ["age", "sibsp", "parch", "fare", "family_size", "is_child",
            "fare_per_person", "fare_log", "age_x_pclass", "pclass"]
cat_cols = ["sex", "embarked", "alone", "who", "adult_male"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=0.1),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

lines = []
best_auc = 0
best_model = None
best_name = ""

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cv = cross_val_score(pipe, X, y, cv=5).mean()
    line = f"{name}: test_acc={acc:.4f} | AUC={auc:.4f} | CV5={cv:.4f}"
    print(line)
    lines.append(line)
    if auc > best_auc:
        best_auc = auc
        best_model = pipe
        best_name = name

joblib.dump({"model": best_model, "features": features},
            "titanic_advanced_model.pkl")

with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write("Titanic advanced results\n")
    f.write(f"features: {features}\n")
    f.write("\n".join(lines) + "\n")
    f.write(f"\nBest (by AUC): {best_name} AUC={best_auc:.4f}\n")

print(f"\nBest: {best_name} -> titanic_advanced_model.pkl + metrics.txt")
