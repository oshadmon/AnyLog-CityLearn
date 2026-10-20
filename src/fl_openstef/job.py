"""
The GUI's OpenSTEF run. The GUI runs in the CityLearn venv (numpy < 2) and OpenSTEF needs numpy 2,
so the GUI starts this file with the OpenSTEF venv's python (`openstef.python` in the config) and
reads its progress from stdout:

  python src/fl_openstef/job.py --config configs/configs.yaml --stop-file <path> [--resume] [--exclude-latest]

Every progress event is one line: @@EVENT {json}. Other lines are ordinary log output.
The run stops after the current community once the stop file exists.
"""
import argparse
import json
import os
import sys
import threading
import time

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pipeline  # noqa: E402
from fl_openstef.backtest import run_backtest  # noqa: E402

PREFIX = "@@EVENT "


def emit(event):
    print(PREFIX + json.dumps(event, default=str), flush=True)


def watch_stop_file(path, stop):
    while not stop.is_set():
        if path and os.path.exists(path):
            stop.set()
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=pipeline.CONFIG_FILE)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude-latest", action="store_true", help="leave out the week live data is still filling")
    args = parser.parse_args()

    settings = pipeline.load_settings(args.config)
    stop = threading.Event()
    threading.Thread(target=watch_stop_file, args=(args.stop_file, stop), daemon=True).start()
    try:
        result = run_backtest(conn=settings["query_node"], settings=settings["openstef"],
                              results_file=settings["openstef_results_file"], on_event=emit, stop=stop,
                              resume=args.resume, exclude_latest=args.exclude_latest)
    except BaseException as error:  # SystemExit from config problems too
        emit({"type": "error", "message": str(error) or error.__class__.__name__})
        raise SystemExit(1)
    if result.get("status") == "no_new_rounds":
        emit({"type": "no_new_rounds", "trained": max([e["round"] for e in result.get("rounds", [])] or [0])})
    emit({"type": "exit", "status": result.get("status", "done")})


if __name__ == "__main__":
    main()