"""
Web GUI backend. Thin on purpose: everything it does comes from the same code the command line
uses (pipeline for publishing, rest for every AnyLog call, fl.server for training).

Three jobs, one of each at a time, each started, stopped and watched the same way:
  insert   set up the operators and publish the simulated history once
  live     keep publishing new hours, timestamped from now
  train    federated training (optionally continuing from the saved model)

  GET  /api/config                  task description, settings, saved model
  GET  /api/network                 sites, the node holding each, assets, row counts
  GET  /api/lookup                  one asset traced from UNS path -> object -> SQL -> rows
  GET  /api/status                  the three jobs' states
  POST /api/jobs/{name}/start       start a job (train takes settings + resume; live takes interval)
  GET  /api/jobs/{name}?since=N     a job's state and its events from N on (the page polls this)
  POST /api/jobs/{name}/stop        stop after the current step
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

import pipeline  # noqa: E402
from rest import build_sql, discover, fetch, node_of, sql_command  # noqa: E402
from fl.server import TASKS, load_model, run_training  # noqa: E402

CONFIG_FILE = os.environ.get("CITYLEARN_CONFIG", pipeline.CONFIG_FILE)


def settings():
    try:
        return pipeline.load_settings(CONFIG_FILE)
    except SystemExit as error:
        raise HTTPException(500, str(error))


def task_for(training):
    name = training.get("task", "load_forecast")
    if name not in TASKS:
        raise HTTPException(400, f"Unknown task {name}. Known: {sorted(TASKS)}")
    return TASKS[name](**training)


class Job:
    """One background job: its state, a stop flag, and the events the page animates from."""
    MAX_EVENTS = 2000  # live data runs for hours; keep the newest events only

    def __init__(self, name):
        self.name, self.lock = name, threading.Lock()
        self.events, self.base = [], 0  # base = events dropped from the front so far
        self.status, self.error, self.info = "idle", None, {}
        self.stop = threading.Event()

    def running(self):
        return self.status in ("running", "stopping")

    def add(self, event):
        with self.lock:
            event["t"] = round(time.time(), 2)
            self.events.append(event)
            extra = len(self.events) - self.MAX_EVENTS
            if extra > 0:
                del self.events[:extra]
                self.base += extra

    def start(self, work, **info):
        with self.lock:
            if self.running():
                raise HTTPException(409, f"{self.name} is already running")
            self.events, self.base, self.status, self.error, self.info = [], 0, "running", None, info
            self.stop = threading.Event()
            stop = self.stop

        def run():
            try:
                status = work(self.add, stop)
                final = status or ("stopped" if stop.is_set() else "done")
            except BaseException as error:  # SystemExit from config problems too
                final = "error"
                self.error = str(error) or error.__class__.__name__
                self.add({"type": "error", "message": self.error})
            with self.lock:
                self.status = final

        threading.Thread(target=run, daemon=True).start()

    def request_stop(self):
        with self.lock:
            if self.status == "running":
                self.stop.set()
                self.status = "stopping"

    def view(self, since=0):
        with self.lock:
            start = max(0, since - self.base)
            return {"name": self.name, "status": self.status, "error": self.error, "info": self.info,
                    "events": self.events[start:], "next": self.base + len(self.events),
                    "last": self.events[-1] if self.events else None}


JOBS = {name: Job(name) for name in ("insert", "live", "train")}
app = FastAPI(title="AnyLog + LF Energy demo")


def throttled(emit, every):
    """Pass on one progress event in `every`, plus the last one of a pass."""
    def on_step(info):
        if info["step"] % every == 0 or info["step"] >= info["steps"] - 1:
            emit(info)
    return on_step


# ---------- the three jobs ----------

def insert_work(emit, stop):
    s = settings()
    emit({"type": "setup", "operators": [o.get("host") for o in s["operators"]]})
    pipeline.setup_operators(s)
    emit({"type": "setup_done"})
    pipeline.publish_history(s, stop=stop, on_step=throttled(emit, 12), setup=False)


def live_work(emit, stop, interval):
    s = settings()
    emit({"type": "setup", "operators": [o.get("host") for o in s["operators"]]})
    pipeline.setup_operators(s)
    pipeline.publish_live(s, stop=stop, interval=interval, setup=False,
                          on_start=lambda week: emit({"type": "live_start", "from_round": week,
                                                      "interval": interval}),
                          on_step=throttled(emit, 1 if interval >= 1 else 6))


def train_work(emit, stop, cfg, resume):
    s = settings()
    live = JOBS["live"].running()
    until = pipeline.complete_rounds(s, live=True) if live else None
    result = run_training(s["query_node"], cfg, s["results_file"], on_event=emit, stop=stop,
                          model_file=s["model_file"], resume=resume, until_round=until)
    return result["status"]


# ---------- endpoints ----------

@app.get("/api/config")
def config():
    s = settings()
    training, content = s["training"], s["content"]
    saved = load_model(s["model_file"])
    return {"query_node": s["query_node"], "uns_root": training.get("uns_root", "citylearn_demo"),
            "training": training, "task": task_for(training).describe(),
            "simulation": {"steps": int((content.get("simulation") or {}).get("steps", 24)),
                           "schema": (content.get("simulation") or {}).get("schema")},
            "steps_per_round": int((content.get("fl") or {}).get("steps_per_round", 168)),
            "live_interval": s["live_interval"],
            "saved_model": {"last_round": saved["last_round"], "saved": saved.get("saved")} if saved else None}


@app.get("/api/network")
def network():
    s = settings()
    query_node, training = s["query_node"], s["training"]
    root = training.get("uns_root", "citylearn_demo")
    train_table = training.get("train_table", "building_obs_train")
    test_table = training.get("test_table", "building_obs_test")
    started = time.time()
    tree = discover(query_node, root)

    sites = []
    for site, leaves in sorted(tree.get(train_table, {}).items()):
        name, address = node_of(leaves[0])
        train = fetch(query_node, leaves, ["COUNT(*) as rows", "MAX(round_number) as rounds"])
        test_leaves = tree.get(test_table, {}).get(site, [])
        test = fetch(query_node, test_leaves, ["COUNT(*) as rows"]) if test_leaves else [{}]
        sites.append({
            "site": site, "node": name, "address": address,
            "assets": [l.name for l in leaves],
            "train_rows": (train[0] if train else {}).get("rows") or 0,
            "test_rows": (test[0] if test else {}).get("rows") or 0,
            "rounds": (train[0] if train else {}).get("rounds") or 0,
        })
    tables = {table: {st: [l.name for l in ls] for st, ls in sorted(by_site.items())}
              for table, by_site in tree.items()}
    return {"query_node": query_node, "uns_root": root, "sites": sites, "tables": tables,
            "train_table": train_table, "test_table": test_table,
            "seconds": round(time.time() - started, 2)}


@app.get("/api/lookup")
def lookup(table: str, site: str, asset: str = "", limit: int = 5):
    """Trace one asset (or a whole site, when asset is empty) from the UNS to its rows."""
    s = settings()
    query_node, training = s["query_node"], s["training"]
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


@app.get("/api/status")
def status():
    return {name: {k: v for k, v in job.view(10 ** 9).items() if k != "events"} for name, job in JOBS.items()}


def job(name):
    if name not in JOBS:
        raise HTTPException(404, f"No job called {name}. Jobs: {sorted(JOBS)}")
    return JOBS[name]


@app.post("/api/jobs/insert/start")
def start_insert():
    if JOBS["live"].running():
        raise HTTPException(409, "Live data is streaming. Stop it before loading the history.")
    if JOBS["train"].running():
        raise HTTPException(409, "Training is running. Stop it before loading the history.")
    JOBS["insert"].start(insert_work)
    return {"status": "running"}


@app.post("/api/jobs/live/start")
def start_live(body: dict = Body(default={})):
    if JOBS["insert"].running():
        raise HTTPException(409, "The history is still loading. Wait for it, or stop it, first.")
    interval = float(body.get("interval") or settings()["live_interval"])
    JOBS["live"].start(lambda emit, stop: live_work(emit, stop, interval), interval=interval)
    return {"status": "running", "interval": interval}


@app.post("/api/jobs/train/start")
def start_train(body: dict = Body(default={})):
    if JOBS["insert"].running():
        raise HTTPException(409, "The history is still loading. Wait for it, or stop it, first.")
    training = settings()["training"]
    task = task_for(training)
    allowed = {s["key"] for s in task.settings} | {"rounds", "history_rounds", "seed"}
    cfg = dict(training, **{k: v for k, v in body.items() if k in allowed and v not in (None, "")})
    resume = bool(body.get("resume"))
    JOBS["train"].start(lambda emit, stop: train_work(emit, stop, cfg, resume), resume=resume)
    return {"status": "running", "config": cfg, "resume": resume}


@app.get("/api/jobs/{name}")
def job_view(name: str, since: int = 0):
    return job(name).view(max(0, since))


@app.post("/api/jobs/{name}/stop")
def job_stop(name: str):
    job(name).request_stop()
    return {"status": job(name).status}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")