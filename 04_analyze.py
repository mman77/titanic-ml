import pandas as pd

df = pd.read_csv("titanic_clean.csv")

# 1. تأثير who
print(df.groupby("who")["survived"].mean())

# 2. تأثير السن - نقسمه فئات
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["child", "teen", "adult", "mid", "old"])
print(df.groupby("age_group", observed=True)["survived"].mean())

# 3. تأثير الأجرة
print(df.groupby("survived")["fare"].describe())
# نقسم الأجرة 4 فئات ونشوف النجاة
df["fare_group"] = pd.qcut(df["fare"], 4)
print(df.groupby("fare_group", observed=True)["survived"].mean())

# 4. تأثير alone و sibsp
print(df.groupby("alone")["survived"].mean())
print(df.groupby("sibsp")["survived"].mean())