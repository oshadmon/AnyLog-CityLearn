"""
Every call to AnyLog, in one place. Everything else (publisher, federated training, OpenSTEF, GUI)
goes through the functions here, and nothing else talks to AnyLog directly.

  connection    configure()  execute()  query_anylog()  rows()
  publishing    publish_policy()  run_msg_client()  publish_to_anylog()  check_data()
  UNS           discover()  list_communities()  get_objects()  node_of()
  row queries   build_sql()  sql_command()  fetch()                  (federated training, GUI)
  hourly        build_query()  query_hourly()  test_windows()        (OpenSTEF)

Settings come from the `anylog` section of configs/configs.yaml (pipeline.load_settings() calls
configure()). Only needs `requests`, so it loads in both venvs.
"""
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Defaults; configure() replaces them from the config file.
CONFIG = {
    "user_agent": "AnyLog/1.23",
    "timeout": 60,
    "topic": "citylearn-demo",                       # REST topic the publisher sends to
    "mapping_policy": "configs/mapping_policy.json",  # relative to the repo root
}

# What OpenSTEF needs per hour: alias -> aggregate. The `openstef.aggregates` config section overrides it.
AGGREGATES = {
    "load": "SUM(net_electricity_consumption)",
    "temperature": "AVG(outdoor_dry_bulb_temperature)",
    "relative_humidity": "AVG(outdoor_relative_humidity)",
    "direct_radiation": "AVG(direct_solar_irradiance)",
    "diffuse_radiation": "AVG(diffuse_solar_irradiance)",
    "price": "AVG(electricity_pricing)",
    "buildings": "COUNT(*)",
}


# ---------------------------------------------------------------- connection

def configure(settings=None):
    """Apply the `anylog` config section (user_agent, timeout, topic, mapping_policy)."""
    CONFIG.update({k: v for k, v in (settings or {}).items() if v is not None})


def execute(conn: str, headers: dict | None = None, payload: str | None = None,
            session: requests.Session | None = None):
    """
    POST one request to an AnyLog node. Returns the parsed JSON reply, or the text when the reply
    isn't JSON. Callers build the headers / payload; query_anylog() covers commands and queries.
    """
    sender = session or requests
    url = conn if conn.startswith("http") else f"http://{conn}"
    try:
        response = sender.request(method="POST", url=url, headers=headers or {}, data=payload,
                                  timeout=CONFIG["timeout"])
        response.raise_for_status()
    except Exception as error:
        command = (headers or {}).get("command") or str(payload)[:120]
        raise IOError(f"AnyLog command failed on {conn}: `{command}` ({error})") from None
    try:
        return response.json()
    except ValueError:
        return response.text


def query_anylog(conn: str, command: str, destination: str | None = None):
    """
    Run one command (a `blockchain get`, `get ...` or `sql ...`) with the command in a JSON body.
    destination="network" for SQL across the operators; None for commands the node answers itself.
    Returns the raw reply; SQL callers pass it through rows().
    """
    body = {"command": command, "AnyLog-Agent": CONFIG["user_agent"]}
    if destination:
        body["destination"] = destination
    return execute(conn=conn, headers={"Content-Type": "application/json"}, payload=json.dumps(body))


def rows(reply):
    """A query reply -> list of rows with lower-case keys. format=json:list returns a list;
    format=json returns {"Query": [...]}."""
    if isinstance(reply, dict):
        reply = reply.get("Query", [])
    if not isinstance(reply, list):
        return []
    return [{str(k).lower(): v for k, v in row.items()} for row in reply]


# ---------------------------------------------------------------- publishing

def mapping_policy(db_name):
    """The mapping policy from its config file, with id (the topic) and dbms filled in."""
    path = CONFIG["mapping_policy"]
    path = path if os.path.isabs(path) else os.path.join(REPO, path)
    with open(path) as f:
        policy = json.load(f)
    policy["mapping"]["id"] = CONFIG["topic"]
    policy["mapping"]["dbms"] = db_name
    return policy


def publish_policy(conn, db_name):
    """Add the mapping policy to the ledger, unless it's there already."""
    if query_anylog(conn, f"blockchain get mapping where id={CONFIG['topic']}"):
        return
    headers = {
        "command": "blockchain insert where policy=!new_policy and local=true and master=!ledger_conn",
        "AnyLog-Agent": CONFIG["user_agent"],
    }
    execute(conn=conn, headers=headers, payload=f"<new_policy={json.dumps(mapping_policy(db_name))}>")


def run_msg_client(conn):
    """Start the REST message client for the topic, unless one is running already."""
    topic = CONFIG["topic"]
    reply = query_anylog(conn, f"get msg client where topic={topic}")
    if str(reply).replace("\n", "").strip() == "No message client subscriptions":
        headers = {
            "command": f"run msg client where broker=rest and log=false and user-agent=anylog and "
                       f"master_node=!ledger_conn and topic=(name={topic}/# and dynamic=true and policy={topic})",
            "AnyLog-Agent": CONFIG["user_agent"],
        }
        execute(conn=conn, headers=headers)


def publish_to_anylog(conn, rows_to_send):
    """
    Send rows to one operator. Each row goes to topic/<table>/<community>/<building>, which the
    message client turns into the UNS path <root>/<table>/<community>/<building>.
    """
    with requests.Session() as session:
        for row in rows_to_send:
            headers = {
                "command": "data",
                "topic": f"{CONFIG['topic']}/{row.get('table')}/{row.get('community_id')}/{row.get('building_id')}",
                "Content-Type": "text/plain",
                "User-Agent": CONFIG["user_agent"],
            }
            execute(conn=conn, headers=headers, payload=json.dumps(row), session=session)


def check_data(conn, db_name, train_table="building_obs_train", test_table="building_obs_test"):
    """Rows per node, table and week, across the network."""
    command = (f'sql {db_name} format=json:list and stat=false and include=({test_table}) and '
               f'extend=(+node_name, @table_name) "SELECT round_number, COUNT(*) FROM {train_table} '
               f'GROUP BY round_number"')
    return rows(query_anylog(conn, command, destination="network"))


# ---------------------------------------------------------------- UNS

@dataclass(frozen=True)
class Leaf:
    namespace: str  # full UNS path
    site: str       # e.g. community_a
    name: str       # e.g. building_1
    dbms: str
    table: str
    where: str
    source_node: str = ""  # operator that created the leaf, e.g. anylog-operator1@172.27.0.4:32148


def node_of(leaf):
    """'anylog-operator1@172.27.0.4:32148' -> ('anylog-operator1', '172.27.0.4:32148')"""
    name, _, address = leaf.source_node.partition("@")
    return name or "unknown", address


def parse_leaves(entries, root, site_level=2):
    """`blockchain get uns ... bring.extend object` output -> leaves that point at data."""
    leaves = []
    for entry in entries if isinstance(entries, list) else []:
        uns, obj = entry.get("uns") or {}, entry.get("object") or {}
        path = str(uns.get("namespace", "")).split("/")
        if path[0] != root or not obj.get("table") or len(path) <= site_level:
            continue  # other trees, and intermediate nodes (their objects carry no table)
        leaves.append(Leaf(namespace=uns["namespace"], site=path[site_level], name=path[-1],
                           dbms=obj["dbms"], table=obj["table"], where=obj.get("where", ""),
                           source_node=uns.get("source_node", "")))
    return leaves


def discover(conn, root, site_level=2):
    """Every leaf under root, in one call -> {table: {site: [Leaf, ...]}}."""
    entries = query_anylog(conn, "blockchain get uns bring.extend object")
    tree = defaultdict(lambda: defaultdict(list))
    for leaf in parse_leaves(entries, root, site_level):
        tree[leaf.table][leaf.site].append(leaf)
    for sites in tree.values():
        for leaves in sites.values():
            leaves.sort(key=lambda l: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", l.namespace)])
    return {table: dict(sites) for table, sites in tree.items()}


def list_communities(conn, root, table):
    """The communities (sites) under root/table in the UNS, in name order."""
    entries = query_anylog(conn, f"blockchain get uns where [namespace] contains {root}/{table}/")
    sites = set()
    for entry in entries if isinstance(entries, list) else []:
        path = str((entry.get("uns") or {}).get("namespace", "")).split("/")
        if len(path) >= 3 and path[0] == root and path[1] == table:
            sites.add(path[2])
    return sorted(sites)


def get_objects(conn, community, root):
    """
    Everything the UNS knows about one community, in one call:
    {table: {"dbms": "mydb", "where": ["building_id = 'Building_13'", ...]}}
    The community's train and test branches both come back, so one query can span both tables.
    """
    entries = query_anylog(conn, f"blockchain get uns where [namespace] contains /{community}/ bring.extend object")
    tables = {}
    for leaf in parse_leaves(entries, root):
        found = tables.setdefault(leaf.table, {"dbms": leaf.dbms, "where": []})
        if leaf.where and leaf.where not in found["where"]:
            found["where"].append(leaf.where)
    if not tables:
        raise LookupError(f"Nothing in the UNS under {root} for {community}")
    return tables


# ---------------------------------------------------------------- row queries (federated training, GUI)

def _or(wheres):
    # AnyLog rejects nested parentheses like ((a) OR (b)), so asset conditions are OR-ed bare in one pair
    return "(" + " OR ".join(wheres) + ")" if wheres else ""


def build_sql(leaves, columns, extra_where=None, limit=None):
    """One query for a site's leaves: their `where`s OR-ed, plus an optional filter."""
    tables = {(l.dbms, l.table) for l in leaves}
    if len(tables) != 1:
        raise ValueError(f"Leaves span several tables: {sorted(tables)}")
    dbms, table = tables.pop()
    conditions = [c for c in (_or([l.where for l in leaves if l.where]), extra_where) if c]
    sql = f"SELECT {', '.join(columns)} FROM {table}" + (f" WHERE {' AND '.join(conditions)}" if conditions else "")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return dbms, sql


def sql_command(dbms, sql, options="format=json and stat=false"):
    """The AnyLog command for a SELECT (shown as is in the GUI)."""
    return f'sql {dbms} {options} "{sql}"'


def fetch(conn, leaves, columns, extra_where=None, limit=None):
    """Rows for one site. AnyLog routes the query to the operator(s) holding the table."""
    dbms, sql = build_sql(leaves, columns, extra_where, limit)
    return rows(query_anylog(conn, sql_command(dbms, sql), destination="network"))


# ---------------------------------------------------------------- hourly aggregates (OpenSTEF)

def _dbms(tables):
    dbms = {t["dbms"] for t in tables.values()}
    if len(dbms) != 1:
        raise ValueError(f"The community's tables span several databases: {sorted(dbms)}")
    return dbms.pop()


def _where(tables, start=None, end=None, column="timestamp"):
    wheres = []
    for t in tables.values():
        wheres += [w for w in t["where"] if w not in wheres]
    conditions = [c for c in (_or(wheres),
                              f"{column} >= '{start}'" if start else "",
                              f"{column} < '{end}'" if end else "") if c]
    return f" WHERE {' AND '.join(conditions)}" if conditions else ""


def build_query(tables, primary=None, start=None, end=None, increments_unit=None, increments_type=None,
                increments_column="timestamp", aggregates=None):
    """
    tables   get_objects() output; dbms, tables and WHERE all come from the UNS
    primary  the table to query FROM (default: the first); the rest go into include=(...)
    start    inclusive, end exclusive, e.g. '2022-08-01 00:00:00'
    """
    primary = primary or sorted(tables)[0]
    others = [t for t in sorted(tables) if t != primary]
    if increments_unit and increments_type:
        time_cols = (f"increments({increments_type}, {increments_unit}, {increments_column}), "
                     f"MIN({increments_column}) AS min_ts, MAX({increments_column}) AS max_ts")
        tail = "ORDER BY min_ts"
    else:
        time_cols = f"{increments_column} AS min_ts"
        tail = f"GROUP BY {increments_column} ORDER BY {increments_column}"
    columns = ", ".join(f"{expr} AS {alias}" for alias, expr in (aggregates or AGGREGATES).items())
    sql = f"SELECT {time_cols}, {columns} FROM {primary}{_where(tables, start, end, increments_column)} {tail}"
    options = "format=json:list and stat=false"
    if others:
        options += f" and include=({','.join(others)}) and extend=(@table_name)"
    return sql_command(_dbms(tables), sql, options)


def query_hourly(conn, tables, primary=None, start=None, end=None, aggregates=None):
    """One community's hourly aggregates for start <= timestamp < end, across all its tables."""
    command = build_query(tables, primary=primary, start=start, end=end, increments_unit=1,
                          increments_type="hour", aggregates=aggregates)
    return rows(query_anylog(conn, command, destination="network"))


def test_windows(conn, tables, test_table):
    """
    Each week's held-back window, from the test table only:
    [{"round": 1, "start": "2022-08-06 15:00:00", "end": "2022-08-07 23:00:00"}, ...]
    """
    if test_table not in tables:
        raise LookupError(f"No {test_table} objects in the UNS for this community")
    only = {test_table: tables[test_table]}
    sql = (f"SELECT round_number, MIN(timestamp) AS window_start, MAX(timestamp) AS window_end FROM {test_table}"
           f"{_where(only)} GROUP BY round_number ORDER BY round_number")
    found = rows(query_anylog(conn, sql_command(_dbms(only), sql, "format=json:list and stat=false"),
                              destination="network"))
    return [{"round": int(r["round_number"]), "start": str(r["window_start"]), "end": str(r["window_end"])}
            for r in found]