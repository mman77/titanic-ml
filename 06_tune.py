"""Step 6: tuning + حفظ أحسن موديل (الاختيار بالـ CV فقط)."""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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

# الـ imputation جوه الـ Pipeline عشان الـ median/mode يتعلم من الـ train بس
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

best_cv = 0
best_model = None
best_name = ""

for name, (pipe, grid) in grids.items():
    gs = GridSearchCV(pipe, grid, cv=strat_cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    # الاختيار بالـ CV فقط — الـ test بيتلمس مرة واحدة في الآخر
    print(f"{name}: best_params={gs.best_params_} | CV={gs.best_score_:.4f}")
    if gs.best_score_ > best_cv:
        best_cv = gs.best_score_
        best_model = gs.best_estimator_
        best_name = name

# الـ test مرة واحدة للتقارير فقط (مش للاختيار)
test_acc = accuracy_score(y_test, best_model.predict(X_test))

joblib.dump({"model": best_model, "features": features}, "titanic_best_model.pkl")
print(f"\nالأحسن بالـ CV: {best_name} (CV={best_cv:.4f}) | test={test_acc:.4f} -> اتحفظ في titanic_best_model.pkl")
