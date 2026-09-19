import pandas as pd

df = pd.read_csv("titanic.csv")
print("قبل التنظيف:")
print(df.isnull().sum())

# 1. حذف عمود deck لانه 77% ناقص (تنظيف هيكلي فقط)
df = df.drop(columns=["deck"])

# ملحوظة منهجية: ملء القيم الناقصة (age / embarked) اتنقل جوه الـ Pipeline
# في سكريبتات التدريب (SimpleImputer) عشان الـ median/mode يتحسب من الـ train
# بس، وميحصلش تسريب من الـ test. هنا بنسيب الـ NaN زي ما هي عمداً.

print("\nبعد التنظيف الهيكلي (الـ NaN الباقي مقصود وبيتعالج في الـ Pipeline):")
print(df.isnull().sum())

df.to_csv("titanic_clean.csv", index=False)
print("\nاتحفظ في titanic_clean.csv shape:", df.shape)
