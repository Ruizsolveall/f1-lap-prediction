#!/usr/bin/env python3
"""
train_pit.py — Train & evaluate a pit-stop timing classifier with FastF1 data.

It:
  1) loads an F1 session,
  2) extracts per-lap features (speed/tyre/status + simple weather),
  3) builds a label: pit within the next K laps,
  4) trains RandomForestClassifier,
  5) evaluates ROC-AUC & PR-AUC, plots ROC/PR and a simple timeline backtest,
  6) saves model + metrics + figures to an output directory.

Usage (examples):
  python train_pit.py --year 2025 --event "Austrian Grand Prix" --session R --lookahead 2 --outdir outputs/
  python train_pit.py --year 2024 --event "British Grand Prix" --session R --test-size 0.25 --n-estimators 400
"""

import argparse, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fastf1
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)

def enable_cache(cache_dir: str):
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

def extract_driver_lap_features(session) -> pd.DataFrame:
    laps = session.laps.copy()
    total = float(laps['LapNumber'].max())
    rows = []
    for _, lap in laps.iterlaps():
        # Telemetry-based speed aggregates
        try:
            tel = lap.get_car_data()[['Speed']]
            avg_spd = float(tel['Speed'].mean())
            max_spd = float(tel['Speed'].max())
        except Exception:
            avg_spd, max_spd = np.nan, np.nan
        tyre_life = lap.get('TyreLife', np.nan)
        pit_in = int(pd.notna(lap.get('PitInTime', np.nan)))
        lap_frac = float(lap['LapNumber']) / total if total else np.nan
        status = lap.get('TrackStatus', np.nan)
        rows.append({
            'Driver':       lap.get('Driver'),
            'LapNumber':    int(lap.get('LapNumber', 0)),
            'lap_frac':     lap_frac,
            'avg_speed':    avg_spd,
            'max_speed':    max_spd,
            'tyre_life':    float(tyre_life) if pd.notna(tyre_life) else 0.0,
            'pit_in':       pit_in,
            'track_status': str(status) if pd.notna(status) else "0"
        })
    return pd.DataFrame(rows)

def pivot_to_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df.pivot(index='LapNumber', columns='Driver',
                    values=['avg_speed','max_speed','tyre_life']).sort_index()
    base.columns = [f"{feat}_{drv}" for feat, drv in base.columns]
    base = base.fillna(0.0)
    # One-hot track status per lap
    ts = df[['LapNumber','track_status']].drop_duplicates('LapNumber').set_index('LapNumber')
    ts_ohe = pd.get_dummies(ts['track_status'], prefix='status')
    out = base.join(ts_ohe, how='left').fillna(0.0)
    return out

def build_label_any_pit_next_k(df: pd.DataFrame, X_index: pd.Index, K: int) -> pd.Series:
    """Label y[t]=1 if any pit occurs in (t, t+K]"""
    pit_by_lap = (df.groupby('LapNumber')['pit_in'].max() > 0).astype(int)
    y = pd.Series(0, index=X_index, dtype=int)
    for t in X_index:
        fut = range(t+1, t+K+1)
        y.loc[t] = int(pit_by_lap.reindex(fut).fillna(0).sum() > 0)
    return y

def add_simple_weather(session, X: pd.DataFrame) -> pd.DataFrame:
    wdf = session.weather_data.copy()
    if wdf is not None and not wdf.empty:
        X = X.copy()
        X['air_temp']   = float(wdf['AirTemp'].dropna().iloc[0])
        X['track_temp'] = float(wdf['TrackTemp'].dropna().iloc[0])
        X['wind_speed'] = float(wdf['WindSpeed'].dropna().iloc[0])
    else:
        X = X.copy()
        X['air_temp'] = X['track_temp'] = X['wind_speed'] = 0.0
    return X

def plot_and_save_roc_pr(y_true, proba, outdir: Path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)

    # ROC
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--', alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.tight_layout()
    roc_path = outdir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # PR
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.tight_layout()
    pr_path = outdir / "pr_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()

    return str(roc_path), str(pr_path)

def plot_and_save_timeline(test_index, proba, alerts, actual_pit, outdir: Path):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    timeline = pd.DataFrame({
        'prob': proba,
        'alert': alerts,
        'actual_pit': actual_pit.reindex(test_index).fillna(0).astype(int)
    }, index=test_index).sort_index()

    plt.figure(figsize=(11,3))
    plt.plot(timeline.index, timeline['prob'], label='prob(pit in next K)')
    # mark alerts & actual pit events
    al_idx = timeline.index[timeline['alert']==1]
    ap_idx = timeline.index[timeline['actual_pit']==1]
    plt.scatter(al_idx, np.full(len(al_idx), 1.02), label='alert', marker='x')
    plt.scatter(ap_idx, np.full(len(ap_idx), 1.05), label='actual pit', marker='o')
    plt.ylim(0, 1.1); plt.xlabel("Lap"); plt.legend(); plt.title("Alerts vs Actual Pit (test)")
    plt.tight_layout()
    tl_path = outdir / "timeline_test.png"
    plt.savefig(tl_path, dpi=150)
    plt.close()
    return str(tl_path)

def main():
    p = argparse.ArgumentParser(description="Train pit-stop timing classifier (any pit next K laps).")
    p.add_argument("--year", type=int, required=True, help="Season year (e.g., 2025)")
    p.add_argument("--event", type=str, required=True, help='Event name (e.g., "Austrian Grand Prix")')
    p.add_argument("--session", type=str, default="R", help="Session code: R/Q/FP1/FP2/FP3/SQ/SSR (default: R)")
    p.add_argument("--lookahead", type=int, default=2, help="K: pit in the next K laps (default: 2)")
    p.add_argument("--test-size", type=float, default=0.25, help="Test split size (default: 0.25)")
    p.add_argument("--n-estimators", type=int, default=400, help="RandomForest n_estimators (default: 400)")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to save artifacts")
    p.add_argument("--cache", type=str, default="~/.fastf1", help="FastF1 cache directory")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Cache & session
    enable_cache(args.cache)
    session = fastf1.get_session(args.year, args.event, args.session)
    session.load(laps=True, telemetry=True, weather=True)

    # Features
    df = extract_driver_lap_features(session)
    X = pivot_to_lap_features(df)
    X = add_simple_weather(session, X)

    # Label & trim last K laps
    y = build_label_any_pit_next_k(df, X.index, args.lookahead)
    valid = X.index <= (X.index.max() - args.lookahead)
    X, y = X.loc[valid], y.loc[valid]

    # Split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    # Train
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(Xtr, ytr)

    # Predict
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Metrics
    roc = float(roc_auc_score(yte, proba))
    ap  = float(average_precision_score(yte, proba))
    report = classification_report(yte, pred, digits=3)
    cm = confusion_matrix(yte, pred).tolist()

    # Plots
    roc_path, pr_path = plot_and_save_roc_pr(yte, proba, outdir)

    # High-precision alert threshold (optional)
    prec, rec, thr = precision_recall_curve(yte, proba)
    target_precision = 0.8
    best_thr = 0.5
    for pval, r, t in zip(prec, rec, np.r_[thr, 1]):
        if pval >= target_precision:
            best_thr = float(t); break
    alerts = (proba >= best_thr).astype(int)

    # Timeline plot (needs actual pit events on test index)
    pit_by_lap = (df.groupby('LapNumber')['pit_in'].max() > 0).astype(int)
    timeline_path = plot_and_save_timeline(
        test_index=yte.index, proba=proba, alerts=alerts,
        actual_pit=pit_by_lap, outdir=outdir
    )

    # Save artifacts
    try:
        import joblib
        joblib.dump(clf, outdir / "pit_rf_model.joblib")
    except Exception:
        pass

    metrics = {
        "roc_auc": roc,
        "pr_auc": ap,
        "threshold_high_precision": best_thr,
        "confusion_matrix_at_0.5": cm,
        "classification_report_at_0.5": report,
        "figures": {
            "roc_curve": str(roc_path),
            "pr_curve": str(pr_path),
            "timeline_test": str(timeline_path)
        }
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({
        "status": "ok",
        "roc_auc": roc,
        "pr_auc": ap,
        "saved_to": str(outdir.resolve())
    }, indent=2))

if __name__ == "__main__":
    main()
