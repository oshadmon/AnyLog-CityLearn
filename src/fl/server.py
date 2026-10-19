"""
The federated loop. Knows nothing about CityLearn or PyTorch: it asks the Task to train,
combine and evaluate, and asks the UNS where each site's data is.

Each round:
  1. every site trains the shared (federated) state and its own local-only state
  2. the federated states are combined (weighted by rows)
  3. federated, local-only and the task's baseline are scored on every site's test rows

Model reuse: with `model_file`, the states are saved after every round. A later run with
`resume=True` loads them and trains only the rounds that arrived since, so what the first run
learned carries over. A resumed run also trains a "fresh" shared model that starts from scratch
on the same new rounds, so the head start can be measured.

`on_event(dict)` is called at each step (the GUI animates from these); `stop` is an optional
threading.Event checked between steps.
"""
import json
import os
import pickle
import time

from .client import Client
from .data import discover
from .load_forecast import LoadForecastTask

# name in configs.yaml (training.task) -> Task class
TASKS = {LoadForecastTask.name: LoadForecastTask}


def _site_line(site, scores):
    parts = [f"{name} {s.get('mae_kwh', s)}" for name, s in scores.items()]
    return f"  {site:<14} MAE  " + "  ".join(parts)


def _size(state):
    return len(pickle.dumps(state))


def _shapes(state):
    """Shapes of a list of arrays (to check a saved model fits the current settings); None otherwise."""
    try:
        return [tuple(a.shape) for a in state]
    except (TypeError, AttributeError):
        return None


def _write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def make_clients(conn, cfg, task, tree=None):
    root = cfg.get("uns_root", "citylearn_demo")
    train_table = cfg.get("train_table", "building_obs_train")
    test_table = cfg.get("test_table", "building_obs_test")
    tree = tree or discover(conn, root)
    if train_table not in tree:
        raise SystemExit(f"No UNS leaves for {train_table} under {root}. Found: {sorted(tree)}")
    clients = [Client(site, tree[train_table][site], tree.get(test_table, {}).get(site, []),
                      task, conn, cfg.get("history_rounds", 1)) for site in sorted(tree[train_table])]
    for c in clients:
        if not c.test_leaves:
            raise SystemExit(f"{c.site} has no leaves in {test_table}")
    return clients


def rounds_available(conn, configs):
    """The last round every site has data for (0 if there is no data yet)."""
    cfg = dict(configs or {})
    task = TASKS[cfg.get("task", "load_forecast")](**cfg)
    try:
        clients = make_clients(conn, cfg, task)
    except SystemExit:
        return 0
    return min(c.last_round() for c in clients)


def load_model(model_file):
    if model_file and os.path.isfile(model_file):
        with open(model_file, "rb") as f:
            return pickle.load(f)
    return None


def run_training(conn, configs, results_file=None, on_event=None, stop=None,
                 model_file=None, resume=False, until_round=None):
    """
    Train rounds 1..N (or, with resume, the rounds after the saved model's last one).

    model_file   where the states are saved after every round (and loaded from, with resume)
    resume       continue from model_file instead of starting from scratch
    until_round  last round to train (live mode passes the last *complete* round)
    """
    cfg = dict(configs or {})
    task = TASKS[cfg.get("task", "load_forecast")](**cfg)
    emit = on_event or (lambda event: None)
    stopped = lambda: stop is not None and stop.is_set()

    # 1. where is the data? (UNS)
    clients = make_clients(conn, cfg, task)
    sites = {c.site: {"assets": c.assets, "node": c.node[0], "address": c.node[1]} for c in clients}

    # 2. where to start: from scratch, or from the saved model
    initial = task.init_state(cfg.get("seed", 0))
    saved = load_model(model_file) if resume else None
    if resume and saved is None:
        print(f"No saved model at {model_file}; starting from scratch")
    if saved is not None:
        if saved.get("task") != task.name or saved.get("shapes") != _shapes(initial):
            raise SystemExit(f"The saved model in {model_file} was trained with a different task or model size. "
                             f"Delete it, or use the settings it was trained with.")
        first_round = saved["last_round"] + 1
        shared = {"federated": saved["federated"]}
        if cfg.get("compare_fresh", True):  # same new rounds, no head start
            shared["fresh"] = saved.get("fresh") or initial
        local = {c.site: saved["local"].get(c.site, initial) for c in clients}
    else:
        first_round = 1
        shared = {"federated": initial}
        local = {c.site: initial for c in clients}

    # 3. how far: every site must have the round
    available = min(c.last_round() for c in clients)
    last = min(available, int(until_round) if until_round else available)
    if not saved and cfg.get("rounds"):
        last = min(last, int(cfg["rounds"]))
    if first_round > last:
        print(f"No new rounds to train (trained up to {first_round - 1}, data up to {available})")
        return {"status": "no_new_rounds", "last_round": first_round - 1}

    print(f"Training {task.name}: rounds {first_round}-{last} over {len(clients)} sites"
          + (" (continuing from the saved model)" if saved else ""))
    for c in clients:
        print(f"  {c.site}: {len(c.assets)} assets ({', '.join(c.assets)})")

    results = None
    if saved and results_file and os.path.isfile(results_file):
        with open(results_file) as f:
            results = json.load(f)  # keep earlier rounds, so the history reads as one run
    if not results:
        results = {"task": task.describe(), "config": cfg, "sites": sites, "rounds": []}
    results.update({"rounds_planned": last, "status": "running", "sites": sites})
    emit({"type": "start", "rounds": last, "first_round": first_round, "resumed": bool(saved),
          "sites": sites, "task": results["task"]})

    for r in range(first_round, last + 1):
        started = time.time()
        updates, rows, traffic = {name: [] for name in shared}, {}, {}
        for c in clients:
            if stopped():
                break
            emit({"type": "site_training", "round": r, "site": c.site})
            states, n = c.train(r, {**shared, "local": local[c.site]})
            for name in shared:
                updates[name].append((states[name], n))
            local[c.site] = states["local"]
            rows[c.site] = n
            # what stayed at the site vs what was sent to the center
            traffic[c.site] = {"raw_bytes": c.last_rows_bytes, "sent_bytes": _size(states["federated"])}
            emit({"type": "site_trained", "round": r, "site": c.site, "rows": n, **traffic[c.site]})
        if stopped():
            results["status"] = "stopped"
            break

        shared = {name: task.combine(ups) for name, ups in updates.items()}
        emit({"type": "combined", "round": r, "sent_back_bytes": _size(shared["federated"])})

        scores = {c.site: c.evaluate({**shared, "local": local[c.site]}, first_round, last) for c in clients}
        entry = {"round": r, "rows": rows, "traffic": traffic, "scores": scores,
                 "resumed": bool(saved), "seconds": round(time.time() - started, 1)}
        results["rounds"].append(entry)
        emit({"type": "round", **entry})

        print(f"Round {r}/{last}  rows {rows}")
        for site, s in scores.items():
            print(_site_line(site, s))

        if model_file:  # saved every round, so a later run can carry on from here
            os.makedirs(os.path.dirname(os.path.abspath(model_file)), exist_ok=True)
            with open(model_file, "wb") as f:
                pickle.dump({"task": task.name, "shapes": _shapes(initial), "last_round": r,
                             "federated": shared["federated"], "fresh": shared.get("fresh"),
                             "local": local, "saved": time.time()}, f)
        if results_file:
            _write_json(results_file, results)

    if results["status"] == "running":
        results["status"] = "done"
    results["last_round"] = results["rounds"][-1]["round"] if results["rounds"] else first_round - 1
    emit({"type": "done", "status": results["status"]})
    if results_file:
        _write_json(results_file, results)
        print(f"Results saved to {results_file}")
    return results