import argparse
import os
import json

import support
from data_gen_citylearn import data_gen
from rest import run_msg_client, check_data, publish_policy
from fl.server import run_training

DIR_NAME = os.path.dirname(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(DIR_NAME, "configs", "configs.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, type=support.check_configs, default=CONFIG_FILE, help="Configuration  file")
    parser.add_argument("--skip-insert", type=bool, nargs='?', default=False, const=True, help="Skip data insertion")
    parser.add_argument("--train", type=bool, nargs='?', default=False, const=True, help="Execute training of the data")
    args = parser.parse_args()

    content = support.read_yaml(config_file=args.config)
    operators = support.operator_info(content.pop("operators", None))
    if not operators:
        raise SystemExit(f"No operators defined in {args.config}")


    # query node provided in configs great. else use the first operator in the list as query node
    query_node = content.get("query") if content.get("query") else operators[0].get("host")

    if args.skip_insert is False:
        for operator in operators:
            publish_policy(conn=operator.get("host"), db_name=content.get("publish").get("dbms"))
            run_msg_client(conn=operator.get("host"))
        data_gen(operators=operators, configs=content)

    # if "check" in steps:
    data = check_data(conn=query_node, db_name=content.get("publish").get("dbms"))
        # print(json.dumps(data, indent=2))

    if args.train is True and data:
        training = content.get("training") or {}
        results_file = os.path.join(DIR_NAME, training.get("results", "results/fl_results.json"))
        run_training(conn=query_node, configs=training, results_file=results_file)


if __name__ == "__main__":
    main()