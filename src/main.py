"""
  python src/main.py                          publish the simulated history, check it
  python src/main.py --skip-insert --train    train on what's already in AnyLog (from scratch)
  python src/main.py --skip-insert --train --resume
                                              only the weeks added since the last run
  python src/main.py --skip-insert --train --continuous-insert
                                              train on the history, then keep publishing new
                                              (live) weeks and keep training on each one as it
                                              completes, continuing from the last run
  python src/main.py --skip-insert --train --follow
                                              like --continuous-insert, but the live data comes
                                              from another process (e.g. the GUI or a second main.py)

  --training-type fl         federated PyTorch model (default, or `training_type` in the config)
  --training-type openstef   OpenSTEF, one model per community; add --community / --rounds to narrow

Two venvs, because OpenSTEF needs numpy 2 and CityLearn needs numpy < 2:
  CityLearn + PyTorch (requirements.txt)          publishing, live data, --training-type fl, the GUI
  OpenSTEF (requirements-openstef.txt)            --skip-insert --train --training-type openstef
Models and CityLearn are imported only when used, so each venv runs its own part of this file.
"""
import argparse
import threading
import time
import traceback

import pipeline
import support
from rest import check_data

TRAINING_TYPES = ("fl", "openstef")


def train(settings, training_type, resume=False, live=False, communities=None, rounds=None):
    """One training run of either kind. live=True leaves out the week still being filled."""
    query_node = settings["query_node"]
    if training_type == "fl":
        from fl.server import run_training

        return run_training(conn=query_node, configs=settings["training"], results_file=settings["results_file"],
                            model_file=settings["model_file"], resume=resume,
                            until_round=pipeline.complete_rounds(settings, live=True) if live else None)

    from fl_openstef.backtest import run_backtest

    cfg = dict(settings["openstef"])
    if rounds:
        cfg["rounds"] = rounds
    return run_backtest(conn=query_node, settings=cfg, results_file=settings["openstef_results_file"],
                        communities=communities, resume=resume, exclude_latest=live)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=False, type=support.check_configs, default=pipeline.CONFIG_FILE, help="Configuration  file")
    parser.add_argument("--skip-insert", type=bool, nargs='?', default=False, const=True, help="Skip publishing the simulated history")
    parser.add_argument("--train", type=bool, nargs='?', default=False, const=True, help="Train on the data in AnyLog")
    parser.add_argument("--resume", type=bool, nargs='?', default=False, const=True, help="Only the weeks added since the last run")
    parser.add_argument("--continuous-insert", type=bool, nargs='?', default=False, const=True, help="Keep publishing new weeks (live), and with --train keep training on them")
    parser.add_argument("--follow", type=bool, nargs='?', default=False, const=True, help="With --train: keep training as new weeks arrive from another publisher")
    parser.add_argument("--training-type", choices=TRAINING_TYPES, default=None, help="fl (PyTorch, federated) or openstef. Default: training_type in the config, else fl")
    parser.add_argument("--community", action="append", help="OpenSTEF: only this community (repeatable)")
    parser.add_argument("--rounds", nargs=2, type=int, metavar=("FIRST", "LAST"), help="OpenSTEF: only these weeks")
    args = parser.parse_args()

    settings = pipeline.load_settings(args.config)
    training_type = args.training_type or settings["training_type"]
    if training_type not in TRAINING_TYPES:
        raise SystemExit(f"Unknown training type {training_type!r}. Choose from {TRAINING_TYPES}")

    # 1. history
    if args.skip_insert is False:
        pipeline.publish_history(settings)

    data = check_data(conn=settings["query_node"], db_name=settings["db_name"],
                      train_table=settings["training"].get("train_table", "building_obs_train"),
                      test_table=settings["training"].get("test_table", "building_obs_test"))
    if args.train and not data:
        raise SystemExit("No data in AnyLog to train on. Publish the history first.")

    # 2. train on what's there
    if args.train:
        train(settings, training_type, resume=args.resume, communities=args.community, rounds=args.rounds)

    if not (args.continuous_insert or args.follow):
        return

    # 3. live: publish new weeks in the background (continuing after the last week in AnyLog),
    #    and/or keep training as each new week completes
    stop = threading.Event()
    if args.continuous_insert:
        def live():
            try:
                pipeline.publish_live(settings, stop=stop,
                                      on_start=lambda week: print(f"Live publishing from week {week} on"))
            except Exception:
                print("Live publishing stopped:")
                traceback.print_exc()

        threading.Thread(target=live, daemon=True).start()

    poll = int(settings["training"].get("poll_seconds", 60))
    try:
        while True:
            time.sleep(poll)
            if args.train:
                train(settings, training_type, resume=True, live=True, communities=args.community)
    except KeyboardInterrupt:
        stop.set()
        print("Stopped.")


if __name__ == "__main__":
    main()