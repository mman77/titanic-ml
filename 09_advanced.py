"""Step 9: final honest report (no new model file).

Every model, baselines included, goes through the SAME repeated stratified
CV (5 splits x 3 repeats) on the TRAIN set only, and every estimator uses its
OWN tuned params from best_params.json (written by 06). The winner by
repeated-CV AUC gets a single test evaluation. metrics.txt records seed,
library versions, split details and paired fold-by-fold differences.
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
    METRICS_PATH,
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
if "selected" not in tuned:
    raise SystemExit("best_params.json is in the old format - re-run 06_tune.py")
all_params = tuned["params"]
print("tuned params from 06:", all_params)


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

BASELINES = ("Dummy", "SexRule")
models = {
    "Dummy(most_frequent)": DummyClassifier(strategy="most_frequent"),
    "SexRule": SexRuleClassifier(),
}
for name, build in BUILDERS.items():
    models[name] = Pipeline([("prep", make_preprocess()),
                             ("clf", build(all_params.get(name, {})))])

rcv = make_repeated_cv()
results = {}
for name, model in models.items():
    res = cross_validate(model, X_train, y_train, cv=rcv,
                         scoring=["roc_auc", "accuracy"],
                         n_jobs=None if name.startswith(BASELINES) else -1)
    results[name] = {"auc": res["test_roc_auc"], "acc": res["test_accuracy"]}
    a, c = results[name]["auc"], results[name]["acc"]
    print(f"{name}: CV-AUC={a.mean():.4f} +/- {a.std():.4f}"
          f" | CV-acc={c.mean():.4f} +/- {c.std():.4f}")

real = [n for n in models if not n.startswith(BASELINES)]
best_name = max(real, key=lambda n: results[n]["auc"].mean())
best_auc = results[best_name]["auc"]

# paired comparison on identical folds: how consistent is the winner's edge?
paired_lines = []
for name in models:
    if name == best_name:
        continue
    d = best_auc - results[name]["auc"]
    paired_lines.append(
        f"{best_name} - {name}: mean dAUC={d.mean():+.4f} | "
        f"better in {int((d > 0).sum())}/{len(d)} folds")
print("\n--- paired (same folds) ---")
print("\n".join(paired_lines))

# single test evaluation: winner + SexRule on the same holdout
test_lines = []
for name in (best_name, "SexRule"):
    m = models[name].fit(X_train, y_train)
    pred = m.predict(X_test)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    correct = int((pred == y_test.values).sum())
    test_lines.append(f"{name}: test_acc={acc:.4f} ({correct}/{len(y_test)})"
                      f" | test_AUC={auc:.4f}")
print("\n--- holdout (single look) ---")
print("\n".join(test_lines))

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("Titanic final report (selection by repeated CV on train only)\n")
    f.write(f"seed={RANDOM_STATE} test_size={TEST_SIZE} "
            f"cv=RepeatedStratifiedKFold(5x3) scoring=roc_auc\n")
    f.write(f"sklearn={sklearn.__version__} pandas={pd.__version__}\n")
    f.write(f"features: {FEATURES}\n")
    f.write(f"tuned params (06): {all_params}\n\n")
    for name in models:
        a, c = results[name]["auc"], results[name]["acc"]
        f.write(f"{name}: CV-AUC={a.mean():.4f} +/- {a.std():.4f}"
                f" | CV-acc={c.mean():.4f} +/- {c.std():.4f}\n")
    f.write("\nPaired (same folds):\n" + "\n".join(paired_lines) + "\n")
    f.write(f"\nWinner (by CV-AUC among models): {best_name}\n")
    f.write("Holdout, single look (n=%d):\n" % len(y_test)
            + "\n".join(test_lines) + "\n")
print(f"\nwrote {METRICS_PATH.name}")
