"""Step 9: features جديدة + مقارنة متقدمة + metrics.txt
ملحوظة: داتا seaborn مفيهاش Name/Ticket فـ Title مش متاح،
بنستخدم المتاح: who + fare_per_person + fare_log + age_x_pclass.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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

# الـ imputation جوه الـ Pipeline عشان مفيش تسريب من الـ test
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=0.1),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

lines = []
best_cv = 0
best_model = None
best_name = ""
results = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    # الـ CV على الـ train بس (مش على الداتا كلها) — والاختيار بيه هو كمان
    cv = cross_val_score(pipe, X_train, y_train, cv=strat_cv).mean()
    results[name] = (pipe, cv)
    line = f"{name}: CV5-train={cv:.4f}"
    print(line)
    lines.append(line)
    if cv > best_cv:
        best_cv = cv
        best_model = pipe
        best_name = name

# الـ test مرة واحدة في الآخر للتقارير فقط (مش للاختيار)
pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test, pred)
test_auc = roc_auc_score(y_test, proba)
print(f"\nBest by CV: {best_name} | test_acc={test_acc:.4f} | test_AUC={test_auc:.4f}")

joblib.dump({"model": best_model, "features": features},
            "titanic_advanced_model.pkl")

with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write("Titanic advanced results (selection by CV on train only)\n")
    f.write(f"features: {features}\n")
    f.write("\n".join(lines) + "\n")
    f.write(f"\nBest (by CV): {best_name} CV={best_cv:.4f} | test_acc={test_acc:.4f} | test_AUC={test_auc:.4f}\n")

print(f"\nBest: {best_name} -> titanic_advanced_model.pkl + metrics.txt")
