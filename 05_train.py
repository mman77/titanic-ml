"""Step 5: baselines + simple models with honest reporting.

Reports train accuracy (overfitting check) + stratified CV mean +/- std +
held-out test accuracy. Feature importance uses permutation importance on
unseen data (impurity importance is biased toward continuous columns).
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from common import (
    DATA_CLEAN,
    FEATURES,
    SexRuleClassifier,
    add_features,
    make_cv,
    make_preprocess,
    split,
)

df = pd.read_csv(DATA_CLEAN)
df = add_features(df)
X = df[FEATURES]
y = df["survived"]
X_train, X_test, y_train, y_test = split(X, y)
print("train:", X_train.shape, "test:", X_test.shape)

cv = make_cv()

models = {
    "Dummy(most_frequent)": DummyClassifier(strategy="most_frequent"),
    "SexRule(female->survived)": SexRuleClassifier(),
    "LogisticRegression": Pipeline(
        [("prep", make_preprocess()),
         ("clf", LogisticRegression(max_iter=1000))]),
    "RandomForest": Pipeline(
        [("prep", make_preprocess()),
         ("clf", RandomForestClassifier(n_estimators=200,
                                        random_state=42))]),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n===== {name} =====")
    print(f"train: {train_acc:.4f} | test: {test_acc:.4f}"
          f" (gap: {train_acc - test_acc:+.4f})")
    if name.startswith(("Logistic", "Random")):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=cv)
        print(f"CV5 on train: {scores.mean():.4f} +/- {scores.std():.4f}")

# permutation importance for RF on unseen test data
rf = Pipeline([("prep", make_preprocess()),
               ("clf", RandomForestClassifier(n_estimators=200,
                                              random_state=42))]).fit(X_train, y_train)
r = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
imp = pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
print("\npermutation importance (test):")
print(imp.head(10).to_string())
