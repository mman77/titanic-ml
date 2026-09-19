"""Step 8: تنبؤ لراكب جديد باستخدام الموديل المحفوظ."""
import joblib
import pandas as pd

saved = joblib.load("titanic_best_model.pkl")
model = saved["model"]
features = saved["features"]

def predict_one(pclass=3, sex="male", age=30, sibsp=0, parch=0,
                fare=8.0, embarked="S", alone=True):
    family_size = sibsp + parch + 1
    is_child = int(age < 12)
    row = pd.DataFrame([{
        "pclass": pclass, "sex": sex, "age": age, "sibsp": sibsp,
        "parch": parch, "fare": fare, "embarked": embarked,
        "alone": alone, "family_size": family_size, "is_child": is_child,
    }])[features]
    prob = model.predict_proba(row)[0, 1]
    pred = int(prob >= 0.5)
    return pred, prob

if __name__ == "__main__":
    examples = [
        {"pclass": 1, "sex": "female", "age": 25, "sibsp": 0, "parch": 0,
         "fare": 100, "embarked": "C", "alone": True},
        {"pclass": 3, "sex": "male", "age": 30, "sibsp": 0, "parch": 0,
         "fare": 8.0, "embarked": "S", "alone": True},
        {"pclass": 2, "sex": "female", "age": 8, "sibsp": 1, "parch": 1,
         "fare": 30, "embarked": "S", "alone": False},
    ]
    for ex in examples:
        pred, prob = predict_one(**ex)
        status = "ناجي" if pred == 1 else "غير ناجي"
        print(f"{ex} -> {status} (prob={prob:.2f})")
