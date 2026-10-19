"""
Web GUI backend. Thin on purpose: everything it shows comes from the same code the
command line uses (fl.data for AnyLog, fl.server for training, the Task for wording).

  GET  /api/config            task description + training defaults
  GET  /api/network           sites, the node holding each, assets, row counts
  GET  /api/lookup            one asset traced from UNS path -> object -> SQL -> rows
  POST /api/run               start a training run in the background
  GET  /api/run?since=N       run status and events from N on (the page polls this)
  POST /api/run/stop          stop after the current step
"""
import os
import sys
import threading
import time

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SRC)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import support  # noqa: E402
from fl.data import build_sql, discover, fetch, node_of, sql_command  # noqa: E402
from fl.server import TASKS, run_training  # noqa: E402

CONFIG_FILE = os.environ.get("CITYLEARN_CONFIG", os.path.join(REPO, "configs", "configs.yaml"))


def load_config():
    content = support.read_yaml(CONFIG_FILE) or {}
    operators = support.operator_info(content.get("operators"))
    query_node = content.get("query") or (operators[0].get("host") if operators else None)
    if not query_node:
        raise HTTPException(500, f"No query node or operators in {CONFIG_FILE}")
    return query_node, dict(content.get("training") or {})


def task_for(training):
    name = training.get("task", "load_forecast")
    if name not in TASKS:
        raise HTTPException(400, f"Unknown task {name}. Known: {sorted(TASKS)}")
    return TASKS[name](**training)


class Run:
    """The one training run the GUI can have at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events, self.status, self.error = [], "idle", None
        self.thread, self.stop = None, threading.Event()

    def add(self, event):
        with self.lock:
            event["t"] = round(time.time(), 2)
            self.events.append(event)

    def start(self, conn, cfg, results_file):
        with self.lock:
            if self.status == "running":
                raise HTTPException(409, "A run is already in progress")
            self.events, self.status, self.error = [], "running", None
            self.stop = threading.Event()

        def work():
            try:
                result = run_training(conn, cfg, results_file, on_event=self.add, stop=self.stop)
                self.status = result["status"]
            except BaseException as error:  # SystemExit from config problems too
                self.error, self.status = str(error) or error.__class__.__name__, "error"
                self.add({"type": "error", "message": self.error})

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def view(self, since):
        with self.lock:
            return {"status": self.status, "error": self.error,
                    "events": self.events[since:], "next": len(self.events)}


RUN = Run()
app = FastAPI(title="AnyLog + LF Energy demo")


@app.get("/api/config")
def config():
    query_node, training = load_config()
    task = task_for(training)
    return {"query_node": query_node, "uns_root": training.get("uns_root", "citylearn_demo"),
            "training": training, "task": task.describe()}


@app.get("/api/network")
def network():
    query_node, training = load_config()
    root = training.get("uns_root", "citylearn_demo")
    train_table = training.get("train_table", "building_obs_train")
    test_table = training.get("test_table", "building_obs_test")
    started = time.time()
    tree = discover(query_node, root)
    if train_table not in tree:
        raise HTTPException(404, f"No UNS leaves for {train_table} under {root}")

    sites = []
    for site, leaves in sorted(tree[train_table].items()):
        name, address = node_of(leaves[0])
        train = fetch(query_node, leaves, ["COUNT(*) as rows", "MAX(round_number) as rounds"])
        test_leaves = tree.get(test_table, {}).get(site, [])
        test = fetch(query_node, test_leaves, ["COUNT(*) as rows"]) if test_leaves else [{}]
        sites.append({
            "site": site, "node": name, "address": address,
            "assets": [l.name for l in leaves],
            "train_rows": (train[0] if train else {}).get("rows", 0),
            "test_rows": (test[0] if test else {}).get("rows", 0),
            "rounds": (train[0] if train else {}).get("rounds", 0),
        })
    tables = {table: {s: [l.name for l in ls] for s, ls in sorted(by_site.items())}
              for table, by_site in tree.items()}
    return {"query_node": query_node, "uns_root": root, "sites": sites, "tables": tables,
            "train_table": train_table, "test_table": test_table,
            "seconds": round(time.time() - started, 2)}


@app.get("/api/lookup")
def lookup(table: str, site: str, asset: str = "", limit: int = 5):
    """Trace one asset (or a whole site, when asset is empty) from the UNS to its rows."""
    query_node, training = load_config()
    tree = discover(query_node, training.get("uns_root", "citylearn_demo"))
    leaves = tree.get(table, {}).get(site, [])
    if asset:
        leaves = [l for l in leaves if l.name == asset]
    if not leaves:
        raise HTTPException(404, f"Nothing in the UNS for {table}/{site}/{asset}")

    columns = ["timestamp", "building_id"] + task_for(training).columns
    dbms, sql = build_sql(leaves, columns, limit=limit)
    started = time.time()
    rows = fetch(query_node, leaves, columns, limit=limit)
    count = fetch(query_node, leaves, ["COUNT(*) as rows"])
    name, address = node_of(leaves[0])
    return {
        "leaves": [{"namespace": l.namespace, "path": l.namespace.split("/"),
                    "object": {"dbms": l.dbms, "table": l.table, "where": l.where}} for l in leaves],
        "sql": sql, "command": sql_command(dbms, sql), "query_node": query_node,
        "node": name, "address": address, "rows": rows,
        "total_rows": (count[0] if count else {}).get("rows", 0),
        "ms": round((time.time() - started) * 1000),
    }


@app.post("/api/run")
def start_run(settings: dict = Body(default={})):
    query_node, training = load_config()
    task = task_for(training)
    allowed = {s["key"] for s in task.settings} | {"rounds", "history_rounds", "seed"}
    cfg = dict(training, **{k: v for k, v in settings.items() if k in allowed and v not in (None, "")})
    results_file = os.path.join(REPO, training.get("results", "results/fl_results.json"))
    RUN.start(query_node, cfg, results_file)
    return {"status": "running", "config": cfg}


@app.get("/api/run")
def run_status(since: int = 0):
    return RUN.view(max(0, since))


@app.post("/api/run/stop")
def stop_run():
    RUN.stop.set()
    return {"status": "stopping"}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
