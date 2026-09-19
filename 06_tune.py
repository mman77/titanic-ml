"""Step 6: tune LR / RF / HGB with GridSearchCV (ROC-AUC, stratified CV).

Selection is by CV score only; the test set is touched once at the end.
Saves the best model + sklearn version, and best_params.json for 09.
"""
import json

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from common import (
    DATA_CLEAN,
    FEATURES,
    MODEL_PATH,
    PARAMS_PATH,
    add_features,
    make_cv,
    make_preprocess,
    split,
)

df = add_features(pd.read_csv(DATA_CLEAN))
X = df[FEATURES]
y = df["survived"]
X_train, X_test, y_train, y_test = split(X, y)

grids = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.05, 0.1, 0.5, 1.0]},
    ),
    "RandomForest": (
        # n_estimators fixed large; tune what matters (depth/split)
        RandomForestClassifier(n_estimators=300, random_state=42),
        {"clf__max_depth": [6, 8, None],
         "clf__min_samples_split": [2, 5]},
    ),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(random_state=42),
        {"clf__learning_rate": [0.05, 0.1],
         "clf__max_leaf_nodes": [15, 31]},
    ),
}

cv = make_cv()
best_cv = 0
best_model = None
best_name = ""
best_params = {}

for name, (clf, grid) in grids.items():
    pipe = Pipeline([("prep", make_preprocess()), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"{name}: best_params={gs.best_params_} | CV-AUC={gs.best_score_:.4f}")
    if gs.best_score_ > best_cv:
        best_cv = gs.best_score_
        best_model = gs.best_estimator_
        best_name = name
        best_params = {"model": name,
                       "params": gs.best_params_}

# test touched once, for reporting only
test_pred = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)[:, 1]
print(f"\nBest by CV: {best_name} | test_acc={accuracy_score(y_test, test_pred):.4f}"
      f" | test_AUC={roc_auc_score(y_test, test_proba):.4f}")

joblib.dump({"model": best_model, "features": FEATURES,
             "sklearn_version": sklearn.__version__}, MODEL_PATH)
with open(PARAMS_PATH, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)
print(f"saved {MODEL_PATH.name} (sklearn {sklearn.__version__}) + {PARAMS_PATH.name}")
