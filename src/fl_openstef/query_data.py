import json

from src.rest import __exec_command

# What OpenSTEF needs per hour: alias -> aggregate. Another source (households) passes its own.
AGGREGATES = {
    "load": "SUM(net_electricity_consumption)",
    "temperature": "AVG(outdoor_dry_bulb_temperature)",
    "relative_humidity": "AVG(outdoor_relative_humidity)",
    "direct_radiation": "AVG(direct_solar_irradiance)",
    "diffuse_radiation": "AVG(diffuse_solar_irradiance)",
    "price": "AVG(electricity_pricing)",
    "buildings": "COUNT(*)",
}


def get_objects(conn: str, community: str, root: str = "citylearn_demo"):
    """
    Everything the UNS knows about one community, in one call:
    {table: {"dbms": "mydb", "where": ["building_id = 'Building_13'", ...]}}
    The community's train and test branches both come back, so the query can span both tables.
    """
    headers = {
        "command": f"blockchain get uns where [namespace] contains /{community}/ bring.extend object",
        "AnyLog-Agent": "AnyLog/1.23",
    }
    response = __exec_command(method="POST", conn=conn, headers={"Content-Type": "application/json"},
                              payload=json.dumps(headers))
    tables = {}
    for entry in response.json() or []:
        namespace = (entry.get("uns") or {}).get("namespace", "")
        obj = entry.get("object") or {}
        if not namespace.startswith(f"{root}/") or not obj.get("table"):
            continue  # other UNS trees, and intermediate nodes (their objects carry no table)
        found = tables.setdefault(obj["table"], {"dbms": obj["dbms"], "where": []})
        if obj.get("where") and obj["where"] not in found["where"]:
            found["where"].append(obj["where"])
    if not tables:
        raise LookupError(f"Nothing in the UNS under {root} for {community}")
    return tables


def build_query(tables: dict, primary: str = None, start=None, end=None,
                increments_unit=None, increments_type=None, increments_column="timestamp",
                aggregates: dict = AGGREGATES):
    """
    tables   get_objects() output; dbms, tables and WHERE all come from the UNS
    primary  the table to query FROM (default: the first); the rest go into include=(...)
    start    inclusive, end exclusive, e.g. '2022-08-01 00:00:00'
    """
    dbms = {t["dbms"] for t in tables.values()}
    if len(dbms) != 1:
        raise ValueError(f"The community's tables span several databases: {sorted(dbms)}")
    primary = primary or sorted(tables)[0]
    others = [t for t in sorted(tables) if t != primary]

    # every object's where, once; AnyLog rejects nested parentheses, so they're OR-ed bare
    wheres = []
    for t in tables.values():
        wheres += [w for w in t["where"] if w not in wheres]
    conditions = ["(" + " OR ".join(wheres) + ")"] if wheres else []
    if start:
        conditions.append(f"{increments_column} >= '{start}'")
    if end:
        conditions.append(f"{increments_column} < '{end}'")

    if increments_unit and increments_type:
        time_cols = (f"increments({increments_type}, {increments_unit}, {increments_column}), "
                     f"MIN({increments_column}) AS min_ts, MAX({increments_column}) AS max_ts")
        tail = "ORDER BY min_ts"
    else:
        time_cols = increments_column
        tail = f"GROUP BY {increments_column} ORDER BY {increments_column}"

    columns = ", ".join(f"{expr} AS {alias}" for alias, expr in aggregates.items())
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT {time_cols}, {columns} FROM {primary}{where} {tail}"

    options = "format=json:list and stat=false"
    if others:
        options += f" and include=({','.join(others)}) and extend=(@table_name)"
    return f'sql {dbms.pop()} {options} "{sql}"'


# if __name__ == "__main__":
#     conn = "172.23.160.85:32049"
#     tables = get_objects(conn, "community_c")
#     print({t: v["where"] for t, v in tables.items()})
#     command = build_query(tables, primary="building_obs_train",
#                           start="2022-08-01 00:00:00", end="2022-08-13 15:00:00",
#                           increments_unit=1, increments_type="hour")
#     print(command)
#     headers = {"command": command, "AnyLog-Agent": "AnyLog/1.23", "destination": "network"}
#     response = __exec_command(method="POST", conn=conn, headers={"Content-Type": "application/json"},
#                               payload=json.dumps(headers))
#     rows = response.json()
#     print(len(rows), rows[:2])