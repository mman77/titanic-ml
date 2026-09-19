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

## النتائج الحالية (الاختيار بالـ CV على الـ train فقط — الـ test للتقارير)
- Baseline (05): LR test 82.1% / CV 80.6% | RF test 81.6% / CV 81.3%
- Tuned (06): RF أحسن بالـ CV (0.8146) | test 81.0%
- Advanced (09, features جديدة): RF أحسن بالـ CV (0.8343) | test 81.0% | AUC 0.8495
- ملحوظة: الـ test فيه 179 راكب بس (±5.6%)، ففروق 1-2% بين الموديلات noise
- الموديل المحفوظ: `titanic_best_model.pkl` + `titanic_advanced_model.pkl`
- المقاييس: `metrics.txt`
- الرسومات: `plots/` (confusion_matrix, roc_curve, survival_rates)

## ملاحظات منهجية
- الـ imputation جوه الـ Pipeline (بيتعلم من الـ train بس)
- الـ CV بـ StratifiedKFold وشغال على الـ train فقط

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
