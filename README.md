# Titanic ML (seaborn sample) + Kaggle-ready notebook

Goal: predict survival (binary classification) with an honest methodology:
selection by cross-validation on train only, test touched once.

Data: `titanic.csv` is the seaborn sample (891 rows). The Kaggle competition
files (`train.csv` / `test.csv` with Name/Ticket/Cabin) are used only inside
`titanic.ipynb`, which also extracts `Title`/`Deck` when they exist.

## Run (Windows: `py -3.12 ...`, elsewhere: `python ...`)
```
pip install -r requirements.txt
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
`titanic_clean.csv`, `titanic_best_model.pkl`, `plots/` are regenerated
by the scripts above (not stored in git).

## Results (selection by repeated CV-AUC on train; test n=179, ~±5.6%)
- Dummy (majority): test 61.5%
- SexRule (female -> survived): test 77.7%
- LogisticRegression: CV-AUC 0.855 ± 0.024
- RandomForest (tuned): CV-AUC 0.879 ± 0.026 -> test 78.2% / AUC 0.844
- HistGradientBoosting (tuned): CV-AUC 0.864 ± 0.035
- Full table: `metrics.txt`

## Files
- `common.py` shared seed/paths/features/preprocess/CV/baselines
- `01..04` download / explore / structural clean / analysis (+ 1 plot)
- `05` baselines + models + permutation importance
- `06` GridSearchCV (ROC-AUC) -> model + `best_params.json`
- `07` confusion matrix + ROC + error analysis
- `08` validated single-passenger prediction
- `09` repeated-CV final report -> `metrics.txt`
- `titanic.ipynb` same methodology, Kaggle-ready (writes `submission.csv`)

## Limitations
- n=891 / test=179: 1-2% gaps between models are noise, not signal
- RF overfits train (~98% vs ~78% test); CV is the honest number
- Seaborn sample has no Name/Ticket/Cabin (no Title feature locally)
