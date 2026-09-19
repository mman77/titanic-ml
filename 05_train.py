import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("titanic_clean.csv")

# 1. Feature engineering بسيط
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_child"] = (df["age"] < 12).astype(int)

# 2. نختار الفيتشرز ونتجنب التسريب:
# alive = نسخة من survived (تسريب) / class, who, adult_male, embark_town = مكررين
features = ["pclass", "sex", "age", "sibsp", "parch", "fare",
            "embarked", "alone", "family_size", "is_child"]
X = df[features]
y = df["survived"]

print("features:", features)
print("X shape:", X.shape)

# 3. تقسيم train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("train:", X_train.shape, "test:", X_test.shape)

# 4. تجهيز الأعمدة جوه الـ Pipeline (عشان مفيش تسريب من الـ test):
# رقمية: median imputation (بيتعلم من الـ train بس) + scale
# فئوية: most_frequent imputation + one-hot
num_cols = ["age", "sibsp", "parch", "fare", "family_size", "is_child", "pclass"]
cat_cols = ["sex", "embarked", "alone"]

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
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(pipe, X, y, cv=strat_cv).mean()
    print(f"\n===== {name} =====")
    print(f"test accuracy: {acc:.4f} | CV-5 mean: {cv:.4f}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    # أهم الفيتشرز للـ RandomForest
    if name == "RandomForest":
        feat_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pipe.named_steps["clf"].feature_importances_
        imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        print("top features:")
        print(imp.head(10).to_string())

print("\nخلص التدريب - قارن بين الموديلين واختار الأحسن")
