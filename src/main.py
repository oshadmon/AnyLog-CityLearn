"""
  python src/main.py                          publish the simulated history, check it
  python src/main.py --skip-insert --train    train on what's already in AnyLog (from scratch)
  python src/main.py --skip-insert --train --resume
                                              train only the rounds added since the saved model
  python src/main.py --skip-insert --train --continuous-insert
                                              train on the history, then keep publishing new
                                              (live) rounds and keep training on each one as it
                                              completes, continuing from the saved model
"""
import argparse
import os
import threading
import time
import traceback

import support
from data_gen_citylearn import data_gen
from rest import run_msg_client, check_data, publish_policy
from fl.server import rounds_available, run_training

DIR_NAME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(DIR_NAME, "configs", "configs.yaml")


def setup_operators(operators, db_name):
    """Mapping policy + message client on every operator (both skip what already exists)."""
    for operator in operators:
        publish_policy(conn=operator.get("host"), db_name=db_name)
        run_msg_client(conn=operator.get("host"))


def insert_data(operators, content, repeat=False, start=None, round_offset=0, live=False):
    """One pass over the simulation, or (repeat) pass after pass, each continuing the timestamps
    and round numbers of the one before."""
    while True:
        start, round_offset = data_gen(operators, content, start=start, round_offset=round_offset, live=live)
        if not repeat:
            return start, round_offset


def live_publisher(operators, content, round_offset):
    """Background thread: publish new rounds forever, starting at round_offset + 1, timestamped from now."""
    try:
        insert_data(operators, content, repeat=True, round_offset=round_offset, live=True)
    except Exception:
        print("Live publishing stopped:")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, type=support.check_configs, default=CONFIG_FILE, help="Configuration  file")
    parser.add_argument("--skip-insert", type=bool, nargs='?', default=False, const=True, help="Skip publishing the simulated history")
    parser.add_argument("--train", type=bool, nargs='?', default=False, const=True, help="Train on the data in AnyLog")
    parser.add_argument("--resume", type=bool, nargs='?', default=False, const=True, help="Continue from the saved model instead of starting from scratch")
    parser.add_argument("--continuous-insert", type=bool, nargs='?', default=False, const=True, help="Keep publishing new rounds (live), and with --train keep training on them")
    args = parser.parse_args()

    content = support.read_yaml(config_file=args.config)
    operators = support.operator_info(content.pop("operators", None))
    if not operators:
        raise SystemExit(f"No operators defined in {args.config}")

    db_name = content.get("publish").get("dbms")
    training = content.get("training") or {}
    results_file = os.path.join(DIR_NAME, training.get("results", "results/fl_results.json"))
    model_file = os.path.join(DIR_NAME, training.get("model", "results/fl_model.pkl"))

    # query node provided in configs great. else use the first operator in the list as query node
    query_node = content.get("query") if content.get("query") else operators[0].get("host")

    # 1. history
    if args.skip_insert is False:
        setup_operators(operators, db_name)
        insert_data(operators=operators, content=content)

    data = check_data(conn=query_node, db_name=db_name)

    # 2. train on what's there
    if args.train and data:
        run_training(conn=query_node, configs=training, results_file=results_file,
                     model_file=model_file, resume=args.resume)

    if not args.continuous_insert:
        return

    # 3. live: publish new rounds in the background, continuing after the last round in AnyLog
    setup_operators(operators, db_name)
    last = rounds_available(query_node, training)
    print(f"Live publishing from round {last + 1} on")
    threading.Thread(target=live_publisher, args=(operators, content, last), daemon=True).start()

    poll = int(training.get("poll_seconds", 60))
    try:
        while True:
            time.sleep(poll)
            if args.train:
                # the newest round is still being filled, so train up to the one before it
                complete = rounds_available(query_node, training) - 1
                run_training(conn=query_node, configs=training, results_file=results_file,
                             model_file=model_file, resume=True, until_round=complete)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()