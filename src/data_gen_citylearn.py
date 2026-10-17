"""
Run a CityLearn simulation and publish one row per building per time step into AnyLog.
Buildings are split into communities, one community per AnyLog operator.
"""
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

from citylearn.agents.baseline import BusinessAsUsualAgent
from citylearn.citylearn import CityLearnEnv

from rest import publish_to_anylog


# Columns that every building has. A fixed column set keeps the AnyLog table schema stable
# (different buildings carry different EV / appliance features in the flat vector).
CORE_FEATURES = [
    "month", "day_type", "hour",
    "outdoor_dry_bulb_temperature", "outdoor_relative_humidity",
    "diffuse_solar_irradiance", "direct_solar_irradiance",
    "carbon_intensity", "electricity_pricing",
    "non_shiftable_load", "solar_generation",
    "electrical_storage_soc", "net_electricity_consumption",
]


def make_row(env, i, t, observations, actions, rewards, community_id, start):
    """One building's observation vector -> one row."""
    values = dict(zip(env.observation_names[i], observations[i]))
    acts = dict(zip(env.action_names[i], actions[i]))
    ts = start + timedelta(seconds=t * env.seconds_per_time_step)
    row = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "community_id": community_id,
        "building_id": env.buildings[i].name,
        "time_step": int(t),
    }
    for name in CORE_FEATURES:
        v = values.get(name)
        row[name] = None if v is None else float(v)
    row["action_electrical_storage"] = float(acts.get("electrical_storage", 0.0))
    row["reward"] = float(rewards[i])
    return row


def data_gen(operators: list, configs: dict):
    sim, pub, fl = configs["simulation"], configs["publish"], configs.get("fl", {})

    kwargs = dict(episode_time_steps=sim.get("steps", 24), interface="flat", render_mode="none")
    if sim.get("root"):
        kwargs["root_directory"] = sim["root"]
    env = CityLearnEnv(sim["schema"], **kwargs)
    agent = BusinessAsUsualAgent(env)

    # split buildings across operators in order: 17 over 3 -> 6 / 6 / 5
    owner, k = {}, 0
    size, extra = divmod(len(env.buildings), len(operators))
    for j, op in enumerate(operators):
        cid = op.get("community_id") or f"{pub.get('community_prefix', 'community')}_{j + 1:03d}"
        op["community_id"] = cid[0] if isinstance(cid, list) else cid
        for i in range(k, k + size + (1 if j < extra else 0)):
            owner[i] = op
        k += size + (1 if j < extra else 0)

    observations, _ = env.reset(seed=sim.get("seed", 0))
    if sim.get("start"):
        start = datetime.strptime(str(sim["start"]), "%Y-%m-%d %H:%M:%S")
    else:  # CityLearn only carries month / hour, so anchor on day 1 of that month
        first = dict(zip(env.observation_names[0], observations[0]))
        start = datetime(sim.get("year", 2022), int(first["month"]), 1, int(first["hour"]) % 24)

    fl_on = fl.get("enabled", False)
    label, shift = fl.get("label", "net_electricity_consumption"), fl.get("label_shift", 1)
    spr, test_frac = fl.get("steps_per_round", 168), fl.get("test_fraction", 0.2)

    pending = defaultdict(deque)  # building index -> rows waiting for their label
    sent = defaultdict(int)
    terminated = truncated = False
    while not (terminated or truncated):
        actions = agent.predict(observations)
        t = env.time_step  # the step these observations describe
        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        batches = defaultdict(list)  # (host, table) -> rows
        for i in range(len(env.buildings)):
            op = owner[i]
            row = make_row(env, i, t, observations, actions, rewards, op["community_id"], start)
            if not fl_on:
                batches[(op["host"], pub["table"])].append(row)
                continue
            # label = this same building's value `shift` steps later
            pending[i].append(row)
            if len(pending[i]) > shift:
                r = pending[i].popleft()
                r["label"] = row[label]
                r["round_number"] = r["time_step"] // spr + 1
                r["data_type"] = "test" if r["time_step"] % spr >= spr * (1 - test_frac) else "train"
                r["table"] = f"{pub['table']}_{r['data_type']}"
                batches[(op["host"], f"{pub['table']}_{r['data_type']}")].append(r)

        for (host, table), rows in batches.items():
            publish_to_anylog(host, rows)
            sent[(host, table)] += len(rows)

        observations = next_obs
        if pub.get("interval_seconds"):
            time.sleep(pub["interval_seconds"])

    for (host, table), n in sorted(sent.items()):
        print(f"published {n} rows to {host} {pub['dbms']}.{table}")
    env.close()