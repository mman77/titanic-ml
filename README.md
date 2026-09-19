# Titanic ML

Pipeline كامل من الداتا الخام للموديل المحفوظ.

## التشغيل (استخدم py -3.12 عندك)
```
py -3.12 01_download_data.py
py -3.12 02_explore.py
py -3.12 03_clean.py
py -3.12 04_analyze.py
py -3.12 05_train.py
py -3.12 06_tune.py
py -3.12 07_evaluate.py
py -3.12 08_predict.py
py -3.12 09_advanced.py
```

## النتائج الحالية
- Baseline (05): LR ~82.7% / RF ~81%
- Tuned (06): LR ~82.1% (C=0.1) / RF ~81.6% (300 tree, depth=8)
- Advanced (09, features جديدة): LR **~83.8%** AUC 0.859 | CV 81.8%
- ROC-AUC: ~0.858
- الموديل المحفوظ: `titanic_best_model.pkl` + `titanic_advanced_model.pkl`
- المقاييس: `metrics.txt`
- الرسومات: `plots/` (confusion_matrix, roc_curve, survival_rates)

## الملفات
- `01_download_data.py` تحميل الداتا
- `02_explore.py` استكشاف سريع
- `03_clean.py` تنظيف -> `titanic_clean.csv`
- `04_analyze.py` تحليل groupby
- `05_train.py` baseline (LR + RF)
- `06_tune.py` GridSearch + حفظ أحسن موديل
- `07_evaluate.py` تقييم + رسومات
- `08_predict.py` تنبؤ لراكب جديد
- `09_advanced.py` features جديدة (fare_per_person, fare_log, age_x_pclass, who) + مقارنة + metrics.txt
