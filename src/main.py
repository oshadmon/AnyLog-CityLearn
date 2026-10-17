import argparse
import os


import support
from data_gen_citylearn import data_gen
from rest import run_msg_client, check_data

DIR_NAME = os.path.dirname(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(DIR_NAME, "configs", "configs.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, type=support.check_configs, default=CONFIG_FILE, help="Configuration  file")
    args = parser.parse_args()

    content = support.read_yaml(config_file=args.config)
    operators = support.operator_info(content.pop("operators", None))
    if not operators:
        raise SystemExit(f"No operators defined in {args.config}")

    for operator in operators:
        run_msg_client(conn=operator.get("host"), db_name=content.get("publish").get("dbms"))

    data_gen(operators=operators, configs=content)
    # query node provided in configs great. else use the first operator in the list as query node
    check_data(conn=content.get("query") if CONFIG_FILE.get("query") else operators[0].get("host"),
               db_name=content.get("publish").get("dbms"))

if __name__ == "__main__":
    main()