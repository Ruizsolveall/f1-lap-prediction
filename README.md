# F1 Lap & Pit Stop Prediction with FastF1 and Machine Learning

This project demonstrates how to extract real-time telemetry features from Formula 1 sessions using [FastF1](https://theoehrly.github.io/Fast-F1/), 
and apply machine learning methods for **lap-time regression** and **pit-stop timing prediction**.

---

## Features
- **Lap Time Prediction**
  - Extract per-lap features (average/max speed, tyre life, pit-in flag, track status).
  - Incorporate weather data (air/track temperature, wind speed).
  - Train a `RandomForestRegressor` to predict lap times (in seconds).
  - Evaluate with R², MAE, RMSE and plot Predicted vs Actual.

- **Pit Stop Prediction**
  - Label laps as 1 if a pit occurs within the next *K* laps.
  - Train a `RandomForestClassifier` and evaluate with ROC-AUC and PR-AUC.
  - Support both **session-level** and **per-driver** labels.
  - Threshold tuning for high-precision alerts (few false alarms).
  - Visualise alerts vs actual pit events on a timeline.

- **Advanced Analysis**
  - **Per-driver pit labels**: predict pit timing per driver rather than session-level.
  - **Threshold heatmap**: precision / recall / alerts count vs. threshold.
  - **Feature attribution**: top-k feature importances, SHAP summary (fallback to permutation importance).

---

## Project Structure
- `f1_lap_features.py` – helper functions for feature extraction.
- `train_model.ipynb` – notebook for lap-time regression.
- `pit_stop_prediction.ipynb` – notebook for pit-stop timing prediction (session-level).
- `pit_stop_advanced.ipynb` – notebook with per-driver labels, threshold heatmap, feature attribution.
- `train_pit.py` – CLI script for pit-stop prediction (saves metrics, plots, model).
- `requirements.txt` – dependencies for reproducibility.

---

## Getting Started
```bash
git clone https://github.com/yourusername/f1-lap-prediction.git
cd f1-lap-prediction
pip install -r requirements.txt

Using the CLI Script

The train_pit.py script provides a quick way to run pit-stop prediction from the command line.

Basic usage
python train_pit.py --year 2025 --event "Austrian Grand Prix" --session R --lookahead 2 --outdir outputs/

Options

--year : Season year (e.g. 2025)

--event : Event name (e.g. "British Grand Prix")

--session : Session code (R=Race, Q=Qualifying, FP1/FP2/FP3, SQ, SSR)

--lookahead : Number of laps to look ahead for pit events (default: 2)

--test-size : Fraction of data used for test split (default: 0.25)

--n-estimators : Number of trees in RandomForest (default: 400)

--outdir : Directory to save metrics and figures (default: outputs/)

--cache : FastF1 cache directory (default: ~/.fastf1)

Output

The script saves:

metrics.json — ROC-AUC, PR-AUC, confusion matrix, thresholds

roc_curve.png, pr_curve.png, timeline_test.png — evaluation plots

pit_rf_model.joblib — trained model (if joblib is installed)

Example output:

{
  "status": "ok",
  "roc_auc": 0.85,
  "pr_auc": 0.78,
  "saved_to": "outputs/"
}
