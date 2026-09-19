"""Step 9: final honest report (no new model file).

Compares baselines + the tuned estimators from best_params.json with
repeated stratified CV (5 splits x 3 repeats) on the TRAIN set only.
Winner by repeated-CV AUC gets a single test evaluation. Writes metrics.txt
with seed, library versions, and split details.
"""
import json

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from common import (
    DATA_CLEAN,
    FEATURES,
    PARAMS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    SexRuleClassifier,
    add_features,
    make_preprocess,
    make_repeated_cv,
    split,
)

df = add_features(pd.read_csv(DATA_CLEAN))
X = df[FEATURES]
y = df["survived"]
X_train, X_test, y_train, y_test = split(X, y)

tuned = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
print("tuned selection from 06:", tuned)


def strip(params):
    return {k.replace("clf__", ""): v for k, v in params.items()}


BUILDERS = {
    "LogisticRegression": lambda p: LogisticRegression(
        max_iter=1000, **strip(p)),
    "RandomForest": lambda p: RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, **strip(p)),
    "HistGradientBoosting": lambda p: HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, **strip(p)),
}

models = {
    "Dummy(most_frequent)": DummyClassifier(strategy="most_frequent"),
    "SexRule": SexRuleClassifier(),
}
for name, build in BUILDERS.items():
    params = tuned["params"] if tuned["model"] == name else {}
    clf = build(params)
    models[name] = Pipeline([("prep", make_preprocess()), ("clf", clf)])

rcv = make_repeated_cv()
rows = []
best_auc, best_name = 0, ""
for name, model in models.items():
    if name.startswith(("Dummy", "SexRule")):
        m = model.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        rows.append((name, None, None, acc, auc))
        print(f"{name}: test_acc={acc:.4f} | test_AUC={auc:.4f} (no CV)")
        continue
    cv_res = cross_validate(model, X_train, y_train, cv=rcv,
                            scoring="roc_auc", n_jobs=-1)
    mean, std = cv_res["test_score"].mean(), cv_res["test_score"].std()
    rows.append((name, mean, std, None, None))
    print(f"{name}: repeated-CV AUC={mean:.4f} +/- {std:.4f}")
    if mean > best_auc:
        best_auc, best_name = mean, name

# single test evaluation of the repeated-CV winner
winner = models[best_name].fit(X_train, y_train)
test_acc = accuracy_score(y_test, winner.predict(X_test))
test_auc = roc_auc_score(y_test, winner.predict_proba(X_test)[:, 1])
print(f"\nWinner by repeated-CV: {best_name} | test_acc={test_acc:.4f}"
      f" | test_AUC={test_auc:.4f}")

with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write("Titanic final report (selection by repeated CV on train only)\n")
    f.write(f"seed={RANDOM_STATE} test_size={TEST_SIZE} "
            f"cv=RepeatedStratifiedKFold(5x3) scoring=roc_auc\n")
    f.write(f"sklearn={sklearn.__version__} pandas={pd.__version__}\n")
    f.write(f"features: {FEATURES}\n")
    for name, mean, std, acc, auc in rows:
        if mean is None:
            f.write(f"{name}: test_acc={acc:.4f} | test_AUC={auc:.4f}\n")
        else:
            f.write(f"{name}: CV-AUC={mean:.4f} +/- {std:.4f}\n")
    f.write(f"\nWinner: {best_name} CV-AUC={best_auc:.4f} | "
            f"test_acc={test_acc:.4f} | test_AUC={test_auc:.4f}\n")
print("wrote metrics.txt")
