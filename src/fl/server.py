"""
The federated loop. Knows nothing about CityLearn or PyTorch: it asks the Task to train,
combine and evaluate, and asks the UNS where each site's data is.

Each round:
  1. every site trains the shared (federated) state and its own local-only state
  2. the federated states are combined (weighted by rows)
  3. federated, local-only and the task's baseline are scored on every site's test rows
"""
import json
import os
import time

from .client import Client
from .data import discover
from .load_forecast import LoadForecastTask

TASKS = {LoadForecastTask.name: LoadForecastTask}


def _site_line(site, scores):
    parts = [f"{name} {s.get('mae_kwh', s)}" for name, s in scores.items()]
    return f"  {site:<14} MAE  " + "  ".join(parts)


def run_training(conn, configs, results_file=None):
    cfg = dict(configs or {})
    root = cfg.get("uns_root", "citylearn_demo")
    train_table = cfg.get("train_table", "building_obs_train")
    test_table = cfg.get("test_table", "building_obs_test")
    task = TASKS[cfg.get("task", "load_forecast")](**cfg)

    # 1. where is the data? (UNS)
    tree = discover(conn, root)
    if train_table not in tree:
        raise SystemExit(f"No UNS leaves for {train_table} under {root}. Found: {sorted(tree)}")
    sites = sorted(tree[train_table])
    clients = [Client(site, tree[train_table][site], tree.get(test_table, {}).get(site, []),
                      task, conn, cfg.get("history_rounds", 1)) for site in sites]
    for c in clients:
        print(f"{c.site}: {len(c.assets)} assets ({', '.join(c.assets)})")
        if not c.test_leaves:
            raise SystemExit(f"{c.site} has no leaves in {test_table}")

    # 2. how many rounds? every site must have the round
    rounds = cfg.get("rounds") or min(c.last_round() for c in clients)
    print(f"Training {task.name}: {rounds} rounds over {len(clients)} sites")

    federated = task.init_state(cfg.get("seed", 0))
    local = {c.site: federated for c in clients}  # same start; never combined
    results = {"task": task.name, "uns_root": root, "config": cfg,
               "sites": {c.site: c.assets for c in clients}, "rounds": []}

    for r in range(1, int(rounds) + 1):
        started = time.time()
        updates, rows = [], {}
        for c in clients:
            states, n = c.train(r, {"federated": federated, "local": local[c.site]})
            updates.append((states["federated"], n))
            local[c.site] = states["local"]
            rows[c.site] = n
        federated = task.combine(updates)

        scores = {c.site: c.evaluate({"federated": federated, "local": local[c.site]}) for c in clients}
        results["rounds"].append({"round": r, "rows": rows, "scores": scores,
                                  "seconds": round(time.time() - started, 1)})
        print(f"Round {r}/{rounds}  rows {rows}")
        for site, s in scores.items():
            print(_site_line(site, s))

        if results_file:  # saved every round, so a GUI can follow along
            os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    if results_file:
        print(f"Results saved to {results_file}")
    return results