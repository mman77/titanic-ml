"""Step 6: tuning + حفظ أحسن موديل."""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic_clean.csv")

df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_child"] = (df["age"] < 12).astype(int)

features = ["pclass", "sex", "age", "sibsp", "parch", "fare",
            "embarked", "alone", "family_size", "is_child"]
X = df[features]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ["age", "sibsp", "parch", "fare", "family_size", "is_child", "pclass"]
cat_cols = ["sex", "embarked", "alone"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

grids = {
    "LogisticRegression": (
        Pipeline([("prep", preprocess),
                  ("clf", LogisticRegression(max_iter=1000))]),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        Pipeline([("prep", preprocess),
                  ("clf", RandomForestClassifier(random_state=42))]),
        {"clf__n_estimators": [200, 300],
         "clf__max_depth": [None, 8, 12],
         "clf__min_samples_split": [2, 5]},
    ),
}

best_score = 0
best_model = None
best_name = ""

for name, (pipe, grid) in grids.items():
    gs = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    pred = gs.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name}: best_params={gs.best_params_} | CV={gs.best_score_:.4f} | test={acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_model = gs.best_estimator_
        best_name = name

joblib.dump({"model": best_model, "features": features}, "titanic_best_model.pkl")
print(f"\nالأحسن: {best_name} ({best_score:.4f}) -> اتحفظ في titanic_best_model.pkl")
