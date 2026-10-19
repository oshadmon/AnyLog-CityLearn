"""
The demo's data actions, shared by the command line (main.py) and the GUI (web/app.py):

  setup_operators   mapping policy + message client on every operator
  publish_history   one pass over the simulation (the initial load)
  publish_live      pass after pass, timestamped from now, continuing after the last week in AnyLog
  complete_rounds   the last week every community has in full

Training itself is fl.server.run_training.
"""
import copy
import os

import rest
import support
from rest import publish_policy, run_msg_client

# fl.server (PyTorch) and data_gen_citylearn (CityLearn) are imported where they're used, so this
# module also loads in the OpenSTEF venv, which has neither.

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(REPO, "configs", "configs.yaml")


def load_settings(config_file=CONFIG_FILE):
    """configs.yaml -> everything the actions need, with defaults filled in."""
    content = support.read_yaml(config_file) or {}
    operators = support.operator_info(content.pop("operators", None))
    if not operators:
        raise SystemExit(f"No operators defined in {config_file}")
    rest.configure(content.get("anylog"))  # every AnyLog call goes through rest.py
    training = dict(content.get("training") or {})
    openstef = dict(content.get("openstef") or {})
    for key in ("uns_root", "train_table", "test_table"):  # same data as the federated run
        openstef.setdefault(key, training.get(key))
    openstef_results = openstef.pop("results", "results/openstef_results.json")
    return {
        "content": content,
        "operators": operators,
        "training": training,
        "db_name": (content.get("publish") or {}).get("dbms"),
        # query node provided in configs great. else use the first operator in the list as query node
        "query_node": content.get("query") or operators[0].get("host"),
        "results_file": os.path.join(REPO, training.get("results", "results/fl_results.json")),
        "model_file": os.path.join(REPO, training.get("model", "results/fl_model.pkl")),
        "live_interval": float((content.get("publish") or {}).get("live_interval_seconds", 2)),
        "training_type": content.get("training_type", "fl"),   # fl | openstef
        "openstef": openstef,
        "openstef_results_file": os.path.join(REPO, openstef_results),
    }


def setup_operators(settings):
    """Mapping policy + message client on every operator (both skip what already exists)."""
    for operator in settings["operators"]:
        publish_policy(conn=operator.get("host"), db_name=settings["db_name"])
        run_msg_client(conn=operator.get("host"))


def publish_history(settings, stop=None, on_step=None, setup=True):
    """The initial load: one pass over the simulated period, with the data's own dates."""
    from data_gen_citylearn import data_gen  # imported here so the GUI starts without loading CityLearn

    if setup:
        setup_operators(settings)
    return data_gen(copy.deepcopy(settings["operators"]), settings["content"], stop=stop, on_step=on_step)


def publish_live(settings, stop=None, on_step=None, on_start=None, interval=None, setup=True):
    """
    Publish new hours until `stop` is set, starting after the last week already in AnyLog and
    timestamped from now. Each pass continues the timestamps and week numbers of the one before.
    """
    from data_gen_citylearn import data_gen
    from fl.server import rounds_available

    if setup:
        setup_operators(settings)
    last = rounds_available(settings["query_node"], settings["training"])
    if on_start:
        on_start(last + 1)
    pause = settings["live_interval"] if interval is None else float(interval)
    operators = copy.deepcopy(settings["operators"])
    start, offset = None, last
    while stop is None or not stop.is_set():
        start, offset = data_gen(operators, settings["content"], start=start, round_offset=offset, live=True,
                                 stop=stop, on_step=on_step, interval=pause)


def complete_rounds(settings, live):
    """The last week every community has in full. While live data streams in, the newest week is
    still being filled, so it is left out."""
    from fl.server import rounds_available

    available = rounds_available(settings["query_node"], settings["training"])
    return max(0, available - 1) if live else available