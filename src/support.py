import argparse
import os.path
import yaml

from concurrent.futures import ThreadPoolExecutor


def check_configs(config_file:str):
    full_path = os.path.expanduser(os.path.expandvars(config_file))
    if os.path.isfile(full_path):
        return full_path
    raise argparse.ArgumentTypeError(f"Invalid file {full_path} for configurations - file not found")


def operator_info(content):
    operators = []
    if content:
        for row in content:
            if row.get("community_id") and not isinstance(row.get("community_id"), list):
                row["community_id"] = row["community_id"].split(",")

            operators.append(row)
    return operators





def read_yaml(config_file:str):
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(stream=f)
    except Exception as error:
        raise Exception(f"Failed to read content in {config_file} (Error: {error})")