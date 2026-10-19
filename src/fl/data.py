"""
AnyLog side of the federated loop: find each site's data through the UNS, then fetch it.

The UNS tree is   <root>/<table>/<site>/<asset>   e.g. citylearn_demo/building_obs_train/community_a/building_1
and each leaf's object says where its rows are:   {"dbms": "mydb", "table": "building_obs_train",
"where": "building_id = 'Building_1'"}. Nothing here knows about buildings or CityLearn.
"""
import re
from collections import defaultdict
from dataclasses import dataclass

import requests

HEADERS = {"User-Agent": "AnyLog/1.23"}


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


def anylog_get(conn, command, destination=None, timeout=60):
    headers = dict(HEADERS, command=command)
    if destination:
        headers["destination"] = destination
    url = conn if conn.startswith("http") else f"http://{conn}"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        raise IOError(f"Non-JSON reply to `{command}`: {response.text[:300]}")


def parse_leaves(entries, root, site_level=2):
    """`blockchain get uns bring.extend object` output -> leaves that point at data."""
    leaves = []
    for entry in entries or []:
        uns, obj = entry.get("uns") or {}, entry.get("object") or {}
        path = str(uns.get("namespace", "")).split("/")
        if path[0] != root or not obj.get("table") or len(path) <= site_level:
            continue  # other trees, and intermediate nodes (their objects carry no table)
        leaves.append(Leaf(namespace=uns["namespace"], site=path[site_level], name=path[-1],
                           dbms=obj["dbms"], table=obj["table"], where=obj.get("where", ""),
                           source_node=uns.get("source_node", "")))
    return leaves


def discover(conn, root, site_level=2):
    """-> {table: {site: [Leaf, ...]}} for every leaf under `root`."""
    entries = anylog_get(conn, "blockchain get uns bring.extend object")
    tree = defaultdict(lambda: defaultdict(list))
    for leaf in parse_leaves(entries, root, site_level):
        tree[leaf.table][leaf.site].append(leaf)
    for sites in tree.values():
        for leaves in sites.values():
            leaves.sort(key=lambda l: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", l.namespace)])
    return {table: dict(sites) for table, sites in tree.items()}


def build_sql(leaves, columns, extra_where=None, limit=None):
    """One query for all of a site's leaves: their `where`s OR-ed, plus an optional filter."""
    tables = {(l.dbms, l.table) for l in leaves}
    if len(tables) != 1:
        raise ValueError(f"Leaves span several tables: {sorted(tables)}")
    dbms, table = tables.pop()
    # AnyLog's SQL parser rejects nested parentheses like ((a) OR (b)), so the leaf
    # `where`s are OR-ed bare. Fine for the simple `col = value` filters the UNS generates.
    where = " OR ".join(l.where for l in leaves if l.where)
    if extra_where:
        where = f"({where}) AND {extra_where}" if where else extra_where
    sql = f"SELECT {', '.join(columns)} FROM {table}" + (f" WHERE {where}" if where else "")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return dbms, sql


def sql_command(dbms, sql):
    """The AnyLog command a fetch sends (shown as is in the GUI)."""
    return f'sql {dbms} format=json and stat=false "{sql}"'


def fetch(conn, leaves, columns, extra_where=None, limit=None):
    """Rows for one site. AnyLog routes the query to the operator(s) holding the table."""
    dbms, sql = build_sql(leaves, columns, extra_where, limit)
    reply = anylog_get(conn, sql_command(dbms, sql), destination="network")
    rows = reply.get("Query", []) if isinstance(reply, dict) else []
    return [{str(k).lower(): v for k, v in row.items()} for row in rows]
