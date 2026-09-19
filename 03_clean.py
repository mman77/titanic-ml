import pandas as pd

df = pd.read_csv("titanic.csv")
print("قبل التنظيف:")
print(df.isnull().sum())

# 1. حذف عمود deck لانه 77% ناقص
df = df.drop(columns=["deck"])

# 2. ملء السن بال median حسب الجنس والدرجة
df["age"] = df.groupby(["sex", "pclass"])["age"].transform(
    lambda x: x.fillna(x.median())
)
# لو لسه فيه ناقص املاه بال median العام
df["age"] = df["age"].fillna(df["age"].median())

# 3. ملء embarked بال mode (هما صفين بس)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

print("\nبعد التنظيف:")
print(df.isnull().sum())

df.to_csv("titanic_clean.csv", index=False)
print("\nاتحفظ في titanic_clean.csv shape:", df.shape)
