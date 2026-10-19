"""
The federated loop. Knows nothing about CityLearn or PyTorch: it asks the Task to train,
combine and evaluate, and asks the UNS where each site's data is.

Each round:
  1. every site trains the shared (federated) state and its own local-only state
  2. the federated states are combined (weighted by rows)
  3. federated, local-only and the task's baseline are scored on every site's test rows

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


def run_training(conn, configs, results_file=None, on_event=None, stop=None):
    cfg = dict(configs or {})
    task = TASKS[cfg.get("task", "load_forecast")](**cfg)
    emit = on_event or (lambda event: None)
    stopped = lambda: stop is not None and stop.is_set()

    # 1. where is the data? (UNS)
    clients = make_clients(conn, cfg, task)
    for c in clients:
        print(f"{c.site}: {len(c.assets)} assets ({', '.join(c.assets)})")

    # 2. how many rounds? every site must have the round
    available = min(c.last_round() for c in clients)
    rounds = min(int(cfg.get("rounds") or available), available)
    print(f"Training {task.name}: {rounds} rounds over {len(clients)} sites")

    federated = task.init_state(cfg.get("seed", 0))
    local = {c.site: federated for c in clients}  # same start; never combined
    results = {"task": task.describe(), "config": cfg, "rounds_planned": rounds,
               "sites": {c.site: {"assets": c.assets, "node": c.node[0], "address": c.node[1]} for c in clients},
               "rounds": [], "status": "running"}
    emit({"type": "start", "rounds": rounds, "sites": results["sites"], "task": results["task"]})

    for r in range(1, rounds + 1):
        started = time.time()
        updates, rows, traffic = [], {}, {}
        for c in clients:
            if stopped():
                break
            emit({"type": "site_training", "round": r, "site": c.site})
            states, n = c.train(r, {"federated": federated, "local": local[c.site]})
            updates.append((states["federated"], n))
            local[c.site] = states["local"]
            rows[c.site] = n
            # what stayed at the site vs what was sent to the center
            traffic[c.site] = {"raw_bytes": c.last_rows_bytes, "sent_bytes": _size(states["federated"])}
            emit({"type": "site_trained", "round": r, "site": c.site, "rows": n, **traffic[c.site]})
        if stopped():
            results["status"] = "stopped"
            break

        federated = task.combine(updates)
        emit({"type": "combined", "round": r, "sent_back_bytes": _size(federated)})

        scores = {c.site: c.evaluate({"federated": federated, "local": local[c.site]}) for c in clients}
        entry = {"round": r, "rows": rows, "traffic": traffic, "scores": scores,
                 "seconds": round(time.time() - started, 1)}
        results["rounds"].append(entry)
        emit({"type": "round", **entry})

        print(f"Round {r}/{rounds}  rows {rows}")
        for site, s in scores.items():
            print(_site_line(site, s))

        if results_file:  # saved every round
            os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    if results["status"] == "running":
        results["status"] = "done"
    emit({"type": "done", "status": results["status"]})
    if results_file:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    return results
