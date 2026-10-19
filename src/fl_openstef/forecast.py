"""
OpenSTEF (3.4.x) on AnyLog data: hourly query rows -> OpenSTEF's input frame -> train -> forecast.

Uses OpenSTEF's storage-free "core" pipelines, so nothing is written to MLflow or a database.
"""
import contextlib
import io
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

# OpenSTEF names for the predictors it knows (query alias -> OpenSTEF column)
COLUMNS = {
    "temperature": "temp",
    "relative_humidity": "humidity",
    "price": "day_ahead_electricity_price",
}


def quiet():
    """OpenSTEF logs a warning per lag feature; keep only errors."""
    warnings.filterwarnings("ignore")
    # OpenSTEF builds each logger from its settings (env prefix "openstef_"). Set the level before it
    # loads; importing openstef.settings directly here would hit a circular import inside OpenSTEF.
    os.environ.setdefault("OPENSTEF_LOG_LEVEL", "ERROR")
    if "openstef.settings" in sys.modules:  # already loaded earlier in this process
        sys.modules["openstef.settings"].Settings.log_level = os.environ["OPENSTEF_LOG_LEVEL"]
    try:
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
    except ImportError:
        pass
    for name in ("openstef", "mlflow", "lightgbm", "xgboost"):
        logging.getLogger(name).setLevel(logging.ERROR)


def to_frame(rows, expected_assets=None, max_gap_hours=6):
    """
    Query rows -> OpenSTEF input: UTC hourly index, `load` first, then the predictors.

    expected_assets  drop hours where fewer assets reported (an hour still being filled), or more
                     (the same data loaded twice), so `load` is never a partial or doubled sum
    max_gap_hours    keep only the latest stretch without a longer gap (live data starts long after
                     the history, and lag features across that gap would be meaningless)
    Also returns a short report of what was dropped.
    """
    df = pd.DataFrame(rows)
    report = {"rows": len(df), "incomplete_hours": 0, "segment_dropped_hours": 0}
    if df.empty:
        return pd.DataFrame(columns=["load", *COLUMNS.values(), "radiation"]), report

    if expected_assets and "buildings" in df:
        bad = df["buildings"] != expected_assets
        report["incomplete_hours"] = int(bad.sum())
        df = df[~bad]

    df.index = pd.to_datetime(df["min_ts"], utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    frame = pd.DataFrame(index=df.index)
    frame["load"] = df["load"].astype(float)
    for alias, column in COLUMNS.items():
        if alias in df:
            frame[column] = df[alias].astype(float)
    if "direct_radiation" in df or "diffuse_radiation" in df:
        frame["radiation"] = df.get("direct_radiation", 0).astype(float) + df.get("diffuse_radiation", 0).astype(float)
    if "table_name" in df:
        frame["table_name"] = df["table_name"]

    # latest stretch without a long gap
    gaps = frame.index.to_series().diff() > pd.Timedelta(hours=max_gap_hours)
    if gaps.any():
        start = gaps[gaps].index[-1]
        report["segment_dropped_hours"] = int((frame.index < start).sum())
        frame = frame[frame.index >= start]

    # regular hourly index: missing hours become NaN, which OpenSTEF's validation handles
    if len(frame):
        frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="h", tz="UTC"))
    return frame, report


def prediction_job(name, settings, pid=1):
    from openstef.data_classes.prediction_job import PredictionJobDataClass

    location = {k: float(settings[k]) for k in ("lat", "lon") if settings.get(k) is not None}
    return PredictionJobDataClass(  # no lat/lon -> OpenSTEF's default location (central Netherlands)
        id=pid, name=name, model=settings.get("model", "xgb"), forecast_type="demand",
        horizon_minutes=int(settings.get("horizon_hours", 36)) * 60, resolution_minutes=60,
        quantiles=list(settings.get("quantiles", [0.1, 0.5, 0.9])), **location,
    )


def train(pj, frame):
    """Fit one OpenSTEF model on a community's history. Returns (model, model_specs)."""
    from openstef.data_classes.model_specifications import ModelSpecificationDataClass
    from openstef.pipeline.train_model import train_model_pipeline_core

    data = frame.drop(columns=["table_name"], errors="ignore")
    with contextlib.redirect_stdout(io.StringIO()):  # OpenSTEF fits XGBoost verbosely; keep the console readable
        model, _report, specs, _datasets = train_model_pipeline_core(
            pj, ModelSpecificationDataClass(id=pj["id"]), data, horizons=[0.25, 24.0])
    model.path = "in-memory"  # OpenSTEF reads this when labelling forecasts; normally set by MLflow
    return model, specs


def forecast(pj, model, specs, history, future):
    """
    history  recent hours with known load (OpenSTEF builds its lag features from these)
    future   the hours to forecast, with predictors; their load is ignored
    Returns the forecast for the future hours: forecast, quantile_P10/P50/P90, ...
    """
    from openstef.pipeline.create_forecast import create_forecast_pipeline_core

    future = future.copy()
    future["load"] = np.nan
    data = pd.concat([history, future]).drop(columns=["table_name"], errors="ignore")
    data = data[~data.index.duplicated(keep="first")].sort_index()
    fc = create_forecast_pipeline_core(pj, data, model, specs)
    return fc.loc[fc.index.isin(future.index)]


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    ok = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[ok], predicted[ok]
    if not len(actual):
        return {"mae_kwh": None, "rmse_kwh": None, "r2": None, "samples": 0}
    err = predicted - actual
    ss_tot = float(np.sum((actual - actual.mean()) ** 2)) or 1e-9
    return {"mae_kwh": round(float(np.mean(np.abs(err))), 4),
            "rmse_kwh": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "r2": round(float(1 - np.sum(err ** 2) / ss_tot), 4),
            "samples": int(len(actual))}