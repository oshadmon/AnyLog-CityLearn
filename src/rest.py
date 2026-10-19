import json
import requests
from paho.mqtt import client as mqtt_client

TOPIC="citylearn-demo"


MAPPING_POLICY = {
  "mapping": {
    "id": TOPIC,
    "dbms": None,
    "table": "bring [table]",
    "group_by_attributes": ["building_id"],
    "readings": "",
    "schema": {
      "timestamp": {
        "type": "timestamp",
        "default": "now()",
        "bring": "[timestamp]",
        "optional": False
      },
      "community_id": {
        "type": "string",
        "default": None,
        "bring": "[community_id]",
        "optional": False
      },
      "building_id": {
        "type": "string",
        "default": None,
        "bring": "[building_id]",
        "optional": False
      },
      "time_step": {
        "type": "int",
        "default": None,
        "bring": "[time_step]",
        "optional": True
      },
      "month": {
        "type": "float",
        "default": None,
        "bring": "[month]",
        "optional": True
      },
      "day_type": {
        "type": "float",
        "default": None,
        "bring": "[day_type]",
        "optional": True
      },
      "hour": {
        "type": "float",
        "default": None,
        "bring": "[hour]",
        "optional": True
      },
      "outdoor_dry_bulb_temperature": {
        "type": "float",
        "default": None,
        "bring": "[outdoor_dry_bulb_temperature]",
        "optional": True
      },
      "outdoor_relative_humidity": {
        "type": "float",
        "default": None,
        "bring": "[outdoor_relative_humidity]",
        "optional": True
      },
      "diffuse_solar_irradiance": {
        "type": "float",
        "default": None,
        "bring": "[diffuse_solar_irradiance]",
        "optional": True
      },
      "direct_solar_irradiance": {
        "type": "float",
        "default": None,
        "bring": "[direct_solar_irradiance]",
        "optional": True
      },
      "carbon_intensity": {
        "type": "float",
        "default": None,
        "bring": "[carbon_intensity]",
        "optional": True
      },
      "electricity_pricing": {
        "type": "float",
        "default": None,
        "bring": "[electricity_pricing]",
        "optional": True
      },
      "non_shiftable_load": {
        "type": "float",
        "default": None,
        "bring": "[non_shiftable_load]",
        "optional": True
      },
      "solar_generation": {
        "type": "float",
        "default": None,
        "bring": "[solar_generation]",
        "optional": True
      },
      "electrical_storage_soc": {
        "type": "float",
        "default": None,
        "bring": "[electrical_storage_soc]",
        "optional": True
      },
      "net_electricity_consumption": {
        "type": "float",
        "default": None,
        "bring": "[net_electricity_consumption]",
        "optional": True
      },
      "action_electrical_storage": {
        "type": "float",
        "default": None,
        "bring": "[action_electrical_storage]",
        "optional": True
      },
      "reward": {
        "type": "float",
        "default": None,
        "bring": "[reward]",
        "optional": True
      },
      "label": {
        "type": "float",
        "default": None,
        "bring": "[label]",
        "optional": True
      },
      "round_number": {
        "type": "int",
        "default": None,
        "bring": "[round_number]",
        "optional": False
      },
      "data_type": {
        "type": "string",
        "default": None,
        "bring": "[data_type]",
        "optional": False
      }
    }
  }
}

def __exec_command(method:str, conn:str, headers:dict, payload:str|None=None, timeout:int=30):
    try:
        response = requests.request(method=method.upper(),
                                    url=f"http://{conn}" if not conn.startswith("http") else conn,
                                    headers=headers, data=payload, timeout=timeout)
        response.raise_for_status()
    except Exception as error:
        raise Exception(f"Failed to execute {method.upper()} against {conn} (Error: {error})")
    return response


def publish_policy(conn:str, db_name:str):
    global MAPPING_POLICY

    headers = {
        "command": f"blockchain get mapping where id={TOPIC}",
        "AnyLog-Agent": "AnyLog/1.23"
    }
    response = __exec_command(method="GET", conn=conn, headers=headers)
    if not response.json():
        headers = {
            "command": "blockchain insert where policy=!new_policy and local=true and master=!ledger_conn",
            "AnyLog-Agent": "AnyLog/1.23"
        }
        MAPPING_POLICY["mapping"]["dbms"] = db_name
        __exec_command(method="POST", conn=conn, headers=headers, payload=f"<new_policy={json.dumps(MAPPING_POLICY)}>")


def run_msg_client(conn:str):
    global MAPPING_POLICY

    headers = {
        "command": f"get msg client where topic={TOPIC}",
        "AnyLog-Agent": "AnyLog/1.23"
    }

    response = __exec_command(method="POST", conn=conn, headers={"Content-Type": "application/json"}, payload=json.dumps(headers))
    if response.text.replace('\n', '').strip() == "No message client subscriptions":
        headers = {
            "command": f"run msg client where broker=rest and log=false and user-agent=anylog and master_node=!ledger_conn and topic=(name={TOPIC}/# and dynamic=true and policy={TOPIC})",
            # "command": f"run msg client where broker=local and log=false and master_node=!ledger_conn and topic=(name={TOPIC}/# and dynamic=true and policy={TOPIC})",
            "AnyLog-Agent": "AnyLog/1.23"
        }
        __exec_command(method="POST", conn=conn, headers=headers, payload=f"<new_policy={json.dumps(MAPPING_POLICY)}>")


def publish_to_anylog(conn, rows):
    """AnyLog REST PUT ingestion: the node creates the table from the JSON on first insert."""
    """Sample Data 
    {
        "timestamp": "2022-08-23 08:00:00", "community_id": "community_a", "building_id": "Building_4", 
        "time_step": 536, "month": 8.0, "day_type": 2.0, "hour": 8.0, "outdoor_dry_bulb_temperature": 18.299999237060547, 
        "outdoor_relative_humidity": 84.0, "diffuse_solar_irradiance": 223.0, "direct_solar_irradiance": 79.0, 
        "carbon_intensity": 0.19636766612529755, "electricity_pricing": 0.18483999371528625, 
        "non_shiftable_load": 1.4851332902908325, "solar_generation": 0.5109333395957947, 
        "electrical_storage_soc": 0.0, "net_electricity_consumption": 0.7222266793251038, 
        "action_electrical_storage": 0.0, "reward": 8.560888493686017, "label": 0.9741999506950378, "round_number": 4, 
        "data_type": "train"
    }
    """
    # client = mqtt_client.Client(
    #     callback_api_version=mqtt_client.CallbackAPIVersion.VERSION1,
    #     client_id="python-mqtt-publisher",
    # )

    try:
        # client.connect("172.23.160.85", 32050, keepalive=60)
        # client.loop_start()
        for row in rows:
            # topic = f"citylearn-demo/{row.get('building_id')}"
            # client.publish(topic, json.dumps(row))
            headers = {
                "command": "data",
                "topic": f"{TOPIC}/{row.get('table')}/{row.get('community_id')}/{row.get('building_id')}",
                "Content-Type": "text/plain",
                "User-Agent": "AnyLog/1.23",
            }

            __exec_command(method="POST", conn=conn, headers=headers, payload=json.dumps(row), timeout=30)

        # client.loop_stop()
        # client.disconnect()
    except Exception as error:
        raise Exception(f"Failed to publish (Error: {error})")


def check_data(conn:str, db_name:str):
    """
    :Sample Output:
    [
        {"node_name":"anylog-operator1","table_name":"building_obs_test","round_number":1,"count(*)":198},
        {"node_name":"anylog-operator1","table_name":"building_obs_test","round_number":2,"count(*)":198},
        {"node_name":"anylog-operator1","table_name":"building_obs_test","round_number":3,"count(*)":198},
        {"node_name":"anylog-operator1","table_name":"building_obs_test","round_number":4,"count(*)":186},
        {"node_name":"anylog-operator1","table_name":"building_obs_train","round_number":1,"count(*)":810},
        ...
    ]
    :param conn:
    :param db_name:
    :return:
    """
    query = f'sql {db_name} format=json:list and stat=false and include=(building_obs_test) and  extend=(+node_name, @table_name)  "SELECT round_number, COUNT(*) FROM building_obs_train GROUP BY round_number;"'
    headers = {
        "command": query,
        "AnyLog-Agent": "AnyLog/1.23",
        "destination": "network"
    }

    response = __exec_command(method="POST", conn=conn, headers={"Content-Type": "application/json"}, payload=json.dumps(headers))
    try:
        if response and int(response.status_code) == 200:
            return response.json()
    except Exception as error:
        raise Exception(f"Failed to to execute `{query}` against {conn} (Error: {error})")
