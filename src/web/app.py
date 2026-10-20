"""
Web GUI backend. Thin on purpose: everything it does comes from the same code the command line
uses (pipeline for publishing, rest for every AnyLog call, fl.server for training).

Three jobs, one of each at a time, each started, stopped and watched the same way:
  insert   set up the operators and publish the simulated history once
  live     keep publishing new hours, timestamped from now
  train    federated training (optionally continuing from the saved model), or with
           `training_type: openstef`, OpenSTEF forecasts run in the OpenSTEF venv (fl_openstef/job.py)

  GET  /api/config                  task description, settings, saved model
  GET  /api/network                 sites, the node holding each, assets, row counts
  GET  /api/lookup                  one asset traced from UNS path -> object -> SQL -> rows
  GET  /api/status                  the three jobs' states
  POST /api/jobs/{name}/start       start a job (train takes settings + resume; live takes interval)
  GET  /api/jobs/{name}?since=N     a job's state and its events from N on (the page polls this)
  POST /api/jobs/{name}/stop        stop after the current step
  GET  /api/openstef/results        the OpenSTEF results file (scores and every forecast)
  GET  /api/openstef/query?site=    the AnyLog query behind one community's forecast
"""
import json
import os
import subprocess
import sys
import tempfile
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
from rest import build_query, build_sql, discover, fetch, get_objects, node_of, sql_command  # noqa: E402
from fl.server import TASKS, load_model, run_training  # noqa: E402
from fl_openstef.describe import TASK as OPENSTEF_TASK  # noqa: E402  (text only; OpenSTEF runs in its own venv)

CONFIG_FILE = os.environ.get("CITYLEARN_CONFIG", pipeline.CONFIG_FILE)


TRAINING_TYPES = {"fl": "PyTorch, federated", "openstef": "OpenSTEF"}


def settings(training_type=None):
    """The config, with training_type switched when the page picked another model (the config's is the default)."""
    try:
        s = pipeline.load_settings(CONFIG_FILE)
        s["default_training_type"] = s["training_type"]
        if training_type in TRAINING_TYPES:
            s["training_type"] = training_type
        return s
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


def train_work(emit, stop, cfg, resume, training_type=None):
    s = settings(training_type)
    if s["training_type"] == "openstef":
        return openstef_work(emit, stop, s, resume)
    live = JOBS["live"].running()
    until = pipeline.complete_rounds(s, live=True) if live else None
    result = run_training(s["query_node"], cfg, s["results_file"], on_event=emit, stop=stop,
                          model_file=s["model_file"], resume=resume, until_round=until)
    if result["status"] == "no_new_rounds":
        spr = int((s["content"].get("fl") or {}).get("steps_per_round", 168))
        emit({"type": "no_new_rounds", "trained": result["last_round"], "per_site": result.get("per_site", {}),
              "complete": result.get("complete"), "live": live,
              "progress": live_progress(spr) if live else None})
    return result["status"]


def live_progress(steps_per_round):
    """Where the live stream is in its newest week, and its measured pace (from the live job's events).
    {"week": 5, "hour": 152, "of": 168, "seconds_per_hour": 6.1} or None before the first hour."""
    with JOBS["live"].lock:
        steps = [e for e in JOBS["live"].events if e.get("type") == "step"][-20:]
    if not steps:
        return None
    last = steps[-1]
    pace = None
    same_pass = [e for e in steps if e["step"] <= last["step"]]  # a new pass restarts the step count
    if len(same_pass) >= 2 and same_pass[-1]["step"] > same_pass[0]["step"]:
        pace = round((same_pass[-1]["t"] - same_pass[0]["t"]) / (same_pass[-1]["step"] - same_pass[0]["step"]), 1)
    return {"week": last["round"], "hour": (last["step"] - 1) % steps_per_round + 1, "of": steps_per_round,
            "seconds_per_hour": pace}


def openstef_work(emit, stop, s, resume):
    """Run fl_openstef/job.py with the OpenSTEF venv's python and pass its events on."""
    python = s["openstef_python"]
    problem = pipeline.openstef_problem(python)
    if problem and problem.startswith("packages:"):
        raise RuntimeError(f"{python} is missing {problem.split(':')[1].replace(',', ' and ')}: "
                           f"run `\"{python}\" -m pip install xgboost` (OpenSTEF imports XGBoost for every model type)")
    if not python or not os.path.isfile(python):
        raise RuntimeError("OpenSTEF runs in its own venv. Set openstef.python in configs.yaml to that venv's python "
                           r"(for example C:\...\AnyLog-CityLearn\venv-openstef\Scripts\python.exe)"
                           + (f"; {python} doesn't exist" if python else ""))
    live = JOBS["live"].running()
    stop_file = os.path.join(tempfile.gettempdir(), f"openstef-stop-{os.getpid()}-{time.time_ns()}")
    command = [python, os.path.join(SRC, "fl_openstef", "job.py"), "--config", CONFIG_FILE, "--stop-file", stop_file]
    command += (["--resume"] if resume else []) + (["--exclude-latest"] if live else [])
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)

    def relay_stop():  # the job checks for the stop file between communities
        stop.wait()
        if process.poll() is None:
            open(stop_file, "w").close()
    threading.Thread(target=relay_stop, daemon=True).start()

    status, log, failed = "done", [], None
    for line in process.stdout:
        line = line.rstrip()
        if not line.startswith("@@EVENT "):
            log = (log + [line])[-25:]
            print(f"[openstef] {line}")
            continue
        event = _clean(json.loads(line[len("@@EVENT "):]))
        if event["type"] == "exit":
            status = event["status"]
        elif event["type"] == "error":
            failed = event["message"]
        elif event["type"] == "no_new_rounds":
            spr = int((s["content"].get("fl") or {}).get("steps_per_round", 168))
            status = "no_new_rounds"
            emit({**event, "live": live, "progress": live_progress(spr) if live else None})
        else:
            emit(event)
    process.wait()
    if os.path.exists(stop_file):
        os.remove(stop_file)
    if failed or process.returncode:
        raise RuntimeError(failed or "OpenSTEF stopped unexpectedly:\n" + "\n".join(log[-8:]))
    return "stopped" if stop.is_set() and status != "no_new_rounds" else status


def openstef_results(s):
    path = s["openstef_results_file"]
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return _clean(json.load(f))


def _clean(value):
    """NaN -> None, so older results files still make valid JSON for the page."""
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def openstef_task(s):
    """The task text, with the run's actual settings filled into the specialist details."""
    cfg = {"model": "xgb", "quantiles": [0.1, 0.5, 0.9], "train_days": None, "min_train_days": 14, **s["openstef"]}
    fill = {"model": cfg["model"], "quantiles": ", ".join(f"P{round(q * 100)}" for q in cfg["quantiles"]),
            "train_days": f"last {cfg['train_days']} days" if cfg.get("train_days") else "everything before it",
            "min_train_days": cfg["min_train_days"]}
    return {**OPENSTEF_TASK, "details": [[k, v.format(**fill)] for k, v in OPENSTEF_TASK["details"]]}


# ---------- endpoints ----------

@app.get("/api/config")
def config(type: str | None = None):
    s = settings(type)
    training, content = s["training"], s["content"]
    if s["training_type"] == "openstef":
        done = openstef_results(s) or {}
        weeks = [e["round"] for e in done.get("rounds", [])]
        forecast = [e["round"] for e in done.get("rounds", []) if e.get("scores")]  # skipped weeks have none
        saved = {"last_round": max(weeks), "weeks": len(forecast), "saved": None} if weeks else None
        task = openstef_task(s)
    else:
        saved = load_model(s["model_file"])
        task = task_for(training).describe()
    return {"query_node": s["query_node"], "uns_root": training.get("uns_root", "citylearn_demo"),
            "training_type": s["training_type"], "default_training_type": s["default_training_type"],
            "training_types": TRAINING_TYPES,
            "openstef_ready": not pipeline.openstef_problem(s["openstef_python"]),
            "openstef_problem": pipeline.openstef_problem(s["openstef_python"]),
            "openstef_python": s["openstef_python"],
            "training": training, "task": task,
            "simulation": {"steps": int((content.get("simulation") or {}).get("steps", 24)),
                           "schema": (content.get("simulation") or {}).get("schema")},
            "steps_per_round": int((content.get("fl") or {}).get("steps_per_round", 168)),
            "live_interval": s["live_interval"],
            "saved_model": {"last_round": saved["last_round"], "saved": saved.get("saved"),
                            "weeks": saved.get("weeks")} if saved else None}


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


@app.get("/api/openstef/results")
def openstef_results_view():
    results = openstef_results(settings())
    if results is None:
        raise HTTPException(404, "No OpenSTEF results yet")
    return results


@app.get("/api/fl/results")
def fl_results_view():
    path = settings()["results_file"]
    if not os.path.isfile(path):
        raise HTTPException(404, "No federated results yet")
    with open(path) as f:
        return _clean(json.load(f))


@app.get("/api/openstef/query")
def openstef_query(site: str):
    """The hourly query OpenSTEF's forecast for `site` used (its last week's window), as AnyLog receives it."""
    s = settings()
    cfg = s["openstef"]
    tables = get_objects(s["query_node"], site, cfg.get("uns_root") or "citylearn_demo")
    results = openstef_results(s) or {}
    windows = [e["window"][site] for e in results.get("rounds", []) if site in e.get("window", {})]
    start = end = None
    if windows:
        import datetime
        fmt = "%Y-%m-%d %H:%M:%S"
        first, last = (datetime.datetime.strptime(str(t)[:19], fmt) for t in windows[-1])
        start = (first - datetime.timedelta(days=int(cfg.get("history_days") or 14))).strftime(fmt)
        end = (last + datetime.timedelta(hours=1)).strftime(fmt)
    command = build_query(tables, primary=cfg.get("train_table") or "building_obs_train", start=start, end=end,
                          increments_unit=1, increments_type="hour", aggregates=cfg.get("aggregates"))
    return {"site": site, "command": command, "objects": tables, "query_node": s["query_node"]}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def compare_models(s):
    """Each model's latest results, reduced to what can be compared fairly: how much it beats its own simple guess.
    (The two forecast different things, so their kWh errors can't be compared directly.)"""
    out = {}
    if os.path.isfile(s["results_file"]):
        with open(s["results_file"]) as f:
            fl = _clean(json.load(f))
        if fl.get("rounds"):
            last = fl["rounds"][-1]  # scores of the latest model, on every week's held-back rows
            pairs = [(v.get("federated", {}).get("mae_kwh"), v.get("baseline", {}).get("mae_kwh"))
                     for v in last["scores"].values()]
            pairs = [(m, b) for m, b in pairs if m is not None and b]
            if pairs:
                model, guess = _mean([m for m, _ in pairs]), _mean([b for _, b in pairs])
                out["fl"] = {"label": TRAINING_TYPES["fl"],
                             "forecasts": "each building's use one hour ahead",
                             "guess": "the next hour equals this hour",
                             "unit": "kWh per building-hour", "error": round(model, 3), "guess_error": round(guess, 3),
                             "gain_pct": round(100 * (1 - model / guess)),
                             "wins": sum(m < b for m, b in pairs), "of": len(pairs), "of_what": "communities",
                             "weeks": last["round"]}
    done = openstef_results(s)
    if done and done.get("rounds"):
        pairs = [(v["openstef"].get("mae_kwh"), v["baseline"].get("mae_kwh"))
                 for e in done["rounds"] for v in (e.get("scores") or {}).values()]
        pairs = [(m, b) for m, b in pairs if m is not None and b]
        if pairs:
            model, guess = _mean([m for m, _ in pairs]), _mean([b for _, b in pairs])
            out["openstef"] = {"label": TRAINING_TYPES["openstef"],
                               "forecasts": "each community's total use, up to a day and a half ahead",
                               "guess": "each hour repeats the same hour last week",
                               "unit": "kWh per community-hour", "error": round(model, 3), "guess_error": round(guess, 3),
                               "gain_pct": round(100 * (1 - model / guess)),
                               "wins": sum(m < b for m, b in pairs), "of": len(pairs), "of_what": "community-weeks",
                               "weeks": max(e["round"] for e in done["rounds"])}
    return out


@app.get("/api/compare")
def compare():
    return compare_models(settings())


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
    kind = body.get("type") if body.get("type") in TRAINING_TYPES else settings()["training_type"]
    JOBS["train"].start(lambda emit, stop: train_work(emit, stop, cfg, resume, kind), resume=resume, type=kind)
    return {"status": "running", "config": cfg, "resume": resume, "type": kind}


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