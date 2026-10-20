"""
OpenSTEF short-term forecasts for every community, week by week, on data read from AnyLog.

For each community and each week's held-back window (the last ~20% of the week):
  1. ask the UNS for the community's objects (dbms, tables, WHERE per asset)
  2. query its hourly history up to the window, and train an OpenSTEF model on it
  3. query the recent history plus the window, and forecast the window (up to ~34 h ahead)
  4. score the forecast against the actual load, next to a simple guess: the same hour last week

Each community trains its own model where its data lives; nothing is combined. The weather in the
window is the measured weather, standing in for the weather forecast an operator would use.
"""
import json
import os
import time

import pandas as pd

from . import forecast as fc
from rest import get_objects, list_communities, query_hourly, test_windows

DEFAULTS = {
    "model": "xgb",               # xgb | lgb | linear (OpenSTEF model types)
    "quantiles": [0.1, 0.5, 0.9],
    "horizon_hours": 36,
    "history_days": 14,           # recent history handed to each forecast (lag features go back 14 days)
    "train_days": None,           # None = all history before the window
    "min_train_days": 14,         # skip windows with less history than this
    "lat": None, "lon": None,     # location, for solar-position features
    "rounds": None,               # e.g. [3, 8] to run only weeks 3-8
}

from .describe import TASK  # the GUI text; kept separate so the GUI can read it without OpenSTEF

FMT = "%Y-%m-%d %H:%M:%S"


def _ts(value):
    return pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value)


def _num(value):
    """A float rounded for JSON, or None for missing values (NaN isn't valid JSON)."""
    return None if value is None or pd.isna(value) else round(float(value), 4)


def _write(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def forecast_window(conn, pj, tables, window, settings, n_assets, train_table, train_start=None):
    """Train and forecast one community's one window. Returns a result dict (or a skip reason)."""
    start, end = _ts(window["start"]), _ts(window["end"]) + pd.Timedelta(hours=1)  # end exclusive
    if settings.get("train_days"):
        train_start = start - pd.Timedelta(days=settings["train_days"])

    # 1. history up to the window -> train
    rows = query_hourly(conn, tables, train_table,
                        start=train_start.strftime(FMT) if train_start is not None else None,
                        end=start.strftime(FMT), aggregates=settings.get("aggregates"))
    train_frame, train_report = fc.to_frame(rows, n_assets)
    hours = int(train_frame["load"].notna().sum()) if len(train_frame) else 0
    if hours == 0 and train_report["incomplete_hours"]:
        return {"skipped": f"no usable hours: {train_report['incomplete_hours']} hours don't have exactly one reading "
                           f"per building ({n_assets}); the history may have been loaded more than once"}
    if hours < int(settings["min_train_days"]) * 24:
        return {"skipped": f"only {hours} hours of history (need {int(settings['min_train_days']) * 24})"}
    model, specs = fc.train(pj, train_frame)

    # 2. recent history + the window -> forecast the window
    rows = query_hourly(conn, tables, train_table,
                        start=(start - pd.Timedelta(days=settings["history_days"])).strftime(FMT),
                        end=end.strftime(FMT), aggregates=settings.get("aggregates"))
    frame, report = fc.to_frame(rows, n_assets)
    history, future = frame[frame.index < start], frame[frame.index >= start]
    if future.empty:
        return {"skipped": "no data in the window"}
    forecast = fc.forecast(pj, model, specs, history, future)

    # 3. score
    actual = future["load"].reindex(forecast.index)
    last_week = frame["load"].reindex(forecast.index - pd.Timedelta(days=7)).to_numpy()
    p10, p90 = forecast.get("quantile_P10"), forecast.get("quantile_P90")
    inside = ((actual >= p10) & (actual <= p90)).mean() if p10 is not None and p90 is not None else None
    return {
        "scores": {"openstef": {**fc.metrics(actual, forecast["forecast"]),
                                "p10_p90_coverage": None if inside is None else round(float(inside), 3)},
                   "baseline": fc.metrics(actual, last_week)},
        "train_hours": hours,
        "dropped": {"incomplete_hours": train_report["incomplete_hours"] + report["incomplete_hours"],
                    "before_gap_hours": train_report["segment_dropped_hours"]},
        "forecast": [{"timestamp": ts.strftime(FMT), "actual": _num(a), "forecast": _num(row["forecast"]),
                      "last_week": _num(lw),
                      **{q: _num(row[q]) for q in ("quantile_P10", "quantile_P50", "quantile_P90") if q in row}}
                     for (ts, row), a, lw in zip(forecast.iterrows(), actual, last_week)],
    }


def run_backtest(conn, settings, results_file=None, on_event=None, stop=None, communities=None,
                 resume=False, exclude_latest=False):
    """
    settings        the `openstef` config section, plus uns_root / train_table / test_table
    resume          keep the weeks already in results_file and forecast only the newer ones
    exclude_latest  leave out the newest week (live data is still filling it)
    Returns results shaped like the federated run's: rounds -> scores per community.
    """
    cfg = {**DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None}}
    root = cfg.get("uns_root", "citylearn_demo")
    train_table = cfg.get("train_table", "building_obs_train")
    test_table = cfg.get("test_table", "building_obs_test")
    emit = on_event or (lambda event: None)
    stopped = lambda: stop is not None and stop.is_set()
    fc.quiet()

    # 1. where is the data? (UNS)
    communities = communities or list_communities(conn, root, train_table)
    if not communities:
        raise SystemExit(f"No communities in the UNS under {root}/{train_table}")
    sites, windows = {}, {}
    for pid, community in enumerate(communities, start=1):
        tables = get_objects(conn, community, root)
        if train_table not in tables:
            raise SystemExit(f"{community} has no objects for {train_table}")
        sites[community] = {"tables": tables, "assets": len(tables[train_table]["where"]),
                            "pj": fc.prediction_job(community, cfg, pid)}
        windows[community] = {w["round"]: w for w in test_windows(conn, tables, test_table)}
        print(f"{community}: {sites[community]['assets']} assets, {len(windows[community])} weeks")

    rounds = sorted({r for ws in windows.values() for r in ws})
    if exclude_latest and rounds:
        rounds = rounds[:-1]
    if cfg.get("rounds"):
        first, last = cfg["rounds"]
        rounds = [r for r in rounds if first <= r <= last]

    results = None
    if resume and results_file and os.path.isfile(results_file):
        with open(results_file) as f:
            results = json.load(f)  # keep the weeks already forecast
        done = {e["round"] for e in results.get("rounds", [])}
        rounds = [r for r in rounds if r not in done]
        if not rounds:
            print(f"No new weeks to forecast (done: {sorted(done)})")
            return {**results, "status": "no_new_rounds"}
    if not results:
        results = {"task": TASK, "rounds": [], "forecasts": {}}
    results.update({"config": dict(cfg), "status": "running", "task": TASK,
                    "sites": {s: {"assets": v["assets"], "tables": sorted(v["tables"])} for s, v in sites.items()}})
    for s in sites:
        results["forecasts"].setdefault(s, {})
    emit({"type": "start", "rounds": rounds[-1] if rounds else 0, "first_round": rounds[0] if rounds else 0,
          "sites": results["sites"], "task": TASK, "resumed": bool(resume and results.get("rounds"))})
    print(f"OpenSTEF ({cfg['model']}): weeks {rounds[0] if rounds else '-'}-{rounds[-1] if rounds else '-'} "
          f"over {len(sites)} communities")

    for r in rounds:
        started, scores, skipped = time.time(), {}, {}
        for community, site in sites.items():
            if stopped():
                break
            if r not in windows[community]:
                continue
            emit({"type": "site_training", "round": r, "site": community})
            out = forecast_window(conn, site["pj"], site["tables"], windows[community][r], cfg,
                                  site["assets"], train_table)
            if "skipped" in out:
                skipped[community] = out["skipped"]
                emit({"type": "site_skipped", "round": r, "site": community, "reason": out["skipped"]})
                continue
            scores[community] = out["scores"]
            results["forecasts"][community][str(r)] = out["forecast"]
            emit({"type": "site_trained", "round": r, "site": community, "train_hours": out["train_hours"],
                  "assets": site["assets"], "forecast_hours": len(out["forecast"])})
        if stopped():
            results["status"] = "stopped"
            break
        if scores or skipped:
            entry = {"round": r, "window": {s: [windows[s][r]["start"], windows[s][r]["end"]] for s in sites if r in windows[s]},
                     "scores": scores, "skipped": skipped, "seconds": round(time.time() - started, 1)}
            results["rounds"].append(entry)
            emit({"type": "round", **entry,
                  "forecasts": {s: results["forecasts"][s].get(str(r)) for s in scores}})
            line = "  ".join(f"{s} OpenSTEF {v['openstef']['mae_kwh']} / last week {v['baseline']['mae_kwh']}"
                             for s, v in scores.items())
            print(f"Week {r}: {line or 'skipped (' + '; '.join(sorted(set(skipped.values()))) + ')'}")
        if results_file:
            _write(results_file, results)

    results["rounds"].sort(key=lambda e: e["round"])
    if results["status"] == "running":
        results["status"] = "done"
    emit({"type": "done", "status": results["status"]})
    if results_file:
        _write(results_file, results)
        print(f"Results saved to {results_file}")
    return results