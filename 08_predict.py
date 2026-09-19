"""Step 8: predict for a new passenger using the saved model.

Engineered features are derived via common.add_features (same code as
training -> no train/serve skew). `alone` is derived, not asked.
Inputs are validated: invalid sex/embarked fail loudly instead of
silently predicting garbage (handle_unknown='ignore' would zero them).
age=None is allowed (the pipeline imputes it and sets age_missing=1).
"""
import numpy as np
import pandas as pd

from common import RAW, add_features, load_model

saved = load_model()
model = saved["model"]
features = saved["features"]

VALID_SEX = {"male", "female"}
VALID_EMBARKED = {"S", "C", "Q"}


def predict_one(pclass=3, sex="male", age=30, sibsp=0, parch=0,
                fare=8.0, embarked="S"):
    if sex not in VALID_SEX:
        raise ValueError(f"sex must be one of {VALID_SEX}, got {sex!r}")
    if embarked not in VALID_EMBARKED:
        raise ValueError(
            f"embarked must be one of {VALID_EMBARKED}, got {embarked!r}")
    if pclass not in (1, 2, 3):
        raise ValueError(f"pclass must be 1/2/3, got {pclass!r}")
    if age is not None and not (0 <= age <= 100):
        raise ValueError(f"age out of range: {age!r} (use None if unknown)")
    if sibsp < 0 or parch < 0 or fare < 0:
        raise ValueError("sibsp/parch/fare must be >= 0")

    raw = pd.DataFrame([{
        "pclass": pclass, "sex": sex,
        "age": np.nan if age is None else float(age),
        "sibsp": sibsp, "parch": parch, "fare": float(fare),
        "embarked": embarked,
    }])[RAW]
    row = add_features(raw)[features]
    prob = float(model.predict_proba(row)[0, 1])
    return int(prob >= 0.5), prob


if __name__ == "__main__":
    examples = [
        {"pclass": 1, "sex": "female", "age": 25, "sibsp": 0, "parch": 0,
         "fare": 100, "embarked": "C"},
        {"pclass": 3, "sex": "male", "age": 30, "sibsp": 0, "parch": 0,
         "fare": 8.0, "embarked": "S"},
        {"pclass": 2, "sex": "female", "age": 8, "sibsp": 1, "parch": 1,
         "fare": 30, "embarked": "S"},
        {"pclass": 3, "sex": "male", "age": None, "sibsp": 0, "parch": 0,
         "fare": 8.0, "embarked": "S"},   # unknown age
    ]
    for ex in examples:
        pred, prob = predict_one(**ex)
        status = "survived" if pred == 1 else "died"
        print(f"{ex} -> {status} (prob={prob:.2f})")

    # validation demo (must raise, not silently mispredict)
    try:
        predict_one(sex="Male")
    except ValueError as e:
        print("validation OK:", e)
