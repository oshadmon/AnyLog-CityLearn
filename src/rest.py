import json
import requests

TOPIC="citylearn-demo"


MAPPING_POLICY = {
  "mapping": {
    "id": TOPIC,
    "dbms": "!default_dbms",
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


def run_msg_client(conn:str):
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
        __exec_command(method="POST", conn=conn, headers=headers, payload=f"<new_policy={json.dumps(MAPPING_POLICY)}>")

    headers = {
        "command": f"run msg client where broker=rest and log=false and user-agent=anylog and master_node=!ledger_conn and topic=(name={TOPIC} and dynamic=true and policy={TOPIC})",
        "AnyLog-Agent": "AnyLog/1.23"
    }
    __exec_command(method="POST", conn=conn, headers=headers, payload=f"<new_policy={json.dumps(MAPPING_POLICY)}>")


def publish_to_anylog(conn, rows):
    """AnyLog REST PUT ingestion: the node creates the table from the JSON on first insert."""
    headers = {
        "command": "data",
        "topic": "citylearn-demo",
        "Content-Type": "text/plain",
        "User-Agent": "AnyLog/1.23",
    }

    __exec_command(method="POST", conn=conn, headers=headers,
                   payload=json.dumps(rows) if isinstance(rows, (list, dict)) else rows,
                   timeout=30)


def post_to_anylog(conn:str, headers:dict, payload:str=None):
    return __exec_command(conn=conn, headers=headers, payload=payload)


def msg_client(conn:str):
    """
    {"timestamp": "2022-08-23 08:00:00", "community_id": "community_a", "building_id": "Building_4", "time_step": 536, "month": 8.0, "day_type": 2.0, "hour": 8.0, "outdoor
_dry_bulb_temperature": 18.299999237060547, "outdoor_relative_humidity": 84.0, "diffuse_solar_irradiance": 223.0, "direct_solar_irradiance": 79.0, "carbon_intensity":
0.19636766612529755, "electricity_pricing": 0.18483999371528625, "non_shiftable_load": 1.4851332902908325, "solar_generation": 0.5109333395957947, "electrical_storage_
soc": 0.0, "net_electricity_consumption": 0.7222266793251038, "action_electrical_storage": 0.0, "reward": 8.560888493686017, "label": 0.9741999506950378, "round_number": 4, "data_type": "train"}

    """


    command = """
    run msg client where broker=rest and master_node=!ledger_conn and log=false and topic=(
        name=citylearn-demo and
        dynamic=true and
        dbms=!default_dbms and
        table="bring [table_name]" and
        column.timestamp.timestamp = "bring [timestamp]" and
        column.community_id = (type=string and value="bring [community_id]") and
        column.building_id = (type=string and value="bring [building_id]") and
        column.time_step = (type=int and value="bring [time_step]") and
        column.month = (type=float and value="bring [month]") and
        column.day_type = (type=float and value="bring [day_type]") and
        column.hour = (type=float and value="bring [hour]") and
        column.outdoor_dry_bulb_temperature = (type=float and value="bring [outdoor_dry_bulb_temperature]") and
        column.outdoor_relative_humidity = (type=float and value="bring [outdoor_relative_humidity]") and
        column.diffuse_solar_irradiance = (type=float and value="bring [diffuse_solar_irradiance]") and
        column.direct_solar_irradiance = (type=float and value="bring [direct_solar_irradiance]") and
        column.carbon_intensity = (type=float and value="bring [carbon_intensity]") and
        column.electricity_pricing = (type=float and value="bring [electricity_pricing]") and
        column.non_shiftable_load = (type=float and value="bring [non_shiftable_load]") and
        column.solar_generation = (type=float and value="bring [solar_generation]") and
        column.electrical_storage_soc = (type=float and value="bring [electrical_storage_soc]") and
        column.net_electricity_consumption = (type=float and value="bring [net_electricity_consumption]") and
        column.action_electrical_storage = (type=float and value="bring [action_electrical_storage]") and
        column.reward = (type=float and value="bring [reward]") and
        column.label = (type=float and value="bring [label]") and
        column.round_number = (type=int and value="bring [round_number]") and
        column.data_type = (type=string and value="bring [data_type]")
    )
    """

    headers = {
        "command": command.replace('\n', ' ').strip(),
        "AnyLog-Agent": "AnyLog/1.23"
    }

    __exec_command(conn=conn, headers=headers)


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
    query = f'sql {db_name} format=json:list include=(building_obs_test) and extend=(+node_name, @table_name)  "SELECT round_number, COUNT(*) FROM building_obs_train GROUP BY round_number"'
    headers = {
        "command": query,
        "AnyLog-Agent": "AnyLog/1.23",
        "destination": "network"
    }

    response = post_to_anylog(conn=conn, headers={"Content-Type": "application/json"}, payload=json.dumps(headers))
    try:
        if response and int(response.status_code()) == 200:
            return response.json()
    except Exception as error:
        raise Exception(f"Failed to to execute `{query}` against {conn} (Error: {error})")
