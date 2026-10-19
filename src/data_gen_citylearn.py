"""
Run a CityLearn simulation and publish one row per building per time step into AnyLog.
Buildings are split into communities, one community per AnyLog operator.

One call = one pass over the simulated period. To keep data flowing (live mode), call it again
with the `start` time and `round_offset` the previous call returned, so timestamps and round
numbers carry on instead of starting over.
"""
import datetime
import time
from collections import defaultdict, deque

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
    ts = start + datetime.timedelta(seconds=t * env.seconds_per_time_step)
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


def assign_buildings(env, operators, pub):
    """Split buildings across operators in order: 17 over 3 -> 6 / 6 / 5. Returns {building index: operator}."""
    owner, k = {}, 0
    size, extra = divmod(len(env.buildings), len(operators))
    for j, op in enumerate(operators):
        cid = op.get("community_id") or f"{pub.get('community_prefix', 'community')}_{j + 1:03d}"
        op["community_id"] = cid[0] if isinstance(cid, list) else cid
        n = size + (1 if j < extra else 0)
        for i in range(k, k + n):
            owner[i] = op
        k += n
    return owner


def live_start(first_hour):
    """Now (UTC), rounded down to the most recent time whose hour of day matches the simulation's
    first hour, so a row's timestamp and its `hour` column agree."""
    now = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    start = now.replace(hour=int(first_hour) % 24)
    return start if start <= now else start - datetime.timedelta(days=1)


def data_gen(operators: list, configs: dict, start=None, round_offset: int = 0, live: bool = False,
             stop=None, on_step=None, interval=None):
    """
    Simulate once and publish every row to the operator that owns its building.

    start         timestamp of time_step 0. None = simulation.start from the config, else the
                  1st of the data's first month (historical), or now (live).
    round_offset  added to every round_number, so a second pass continues where the first ended.
    live          only decides the default start (now instead of the data's own dates).
    stop          optional threading.Event; the pass ends early once it is set.
    on_step       optional callback, called after every simulated hour with a progress dict.
    interval      seconds to pause after each simulated hour (default: publish.interval_seconds).

    Returns (start of the next pass, round_offset for the next pass).
    """
    sim, pub, fl = configs["simulation"], configs["publish"], configs.get("fl", {})
    steps = int(sim.get("steps", 24))

    kwargs = dict(episode_time_steps=steps, interface="flat", render_mode="none")
    if sim.get("root"):
        kwargs["root_directory"] = sim["root"]
    env = CityLearnEnv(sim["schema"], **kwargs)
    agent = BusinessAsUsualAgent(env)
    owner = assign_buildings(env, operators, pub)

    observations, _ = env.reset(seed=sim.get("seed", 0))
    first = dict(zip(env.observation_names[0], observations[0]))
    if start is None:
        if live:
            start = live_start(first["hour"])
        elif sim.get("start"):
            start = datetime.datetime.strptime(str(sim["start"]), "%Y-%m-%d %H:%M:%S")
        else:  # CityLearn only carries month / hour, so anchor on day 1 of that month
            start = datetime.datetime(sim.get("year", 2022), int(first["month"]), 1, int(first["hour"]) % 24)

    fl_on = fl.get("enabled", False)
    label, shift = fl.get("label", "net_electricity_consumption"), fl.get("label_shift", 1)
    spr, test_frac = int(fl.get("steps_per_round", 168)), fl.get("test_fraction", 0.2)

    pause = pub.get("interval_seconds") if interval is None else interval
    pending = defaultdict(deque)  # building index -> rows waiting for their label
    sent = defaultdict(int)
    terminated = truncated = False
    while not (terminated or truncated):
        if stop is not None and stop.is_set():
            break
        actions = agent.predict(observations)
        t = env.time_step  # the step these observations describe
        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        batches = defaultdict(list)  # (host, table) -> rows
        for i in range(len(env.buildings)):
            op = owner[i]
            row = make_row(env, i, t, observations, actions, rewards, op["community_id"], start)
            if not fl_on:
                row["table"] = pub["table"]
                batches[(op["host"], pub["table"])].append(row)
                continue
            # label = this same building's value `shift` steps later
            pending[i].append(row)
            if len(pending[i]) > shift:
                r = pending[i].popleft()
                r["label"] = row[label]
                r["round_number"] = round_offset + r["time_step"] // spr + 1
                r["data_type"] = "test" if r["time_step"] % spr >= spr * (1 - test_frac) else "train"
                r["table"] = f"{pub['table']}_{r['data_type']}"
                batches[(op["host"], r["table"])].append(r)

        for (host, table), rows in batches.items():
            publish_to_anylog(host, rows)
            sent[(host, table)] += len(rows)

        if on_step:
            on_step({"type": "step", "step": t + 1, "steps": steps, "round": round_offset + t // spr + 1,
                     "timestamp": (start + datetime.timedelta(seconds=t * env.seconds_per_time_step)).strftime("%Y-%m-%d %H:%M"),
                     "rows": sum(len(r) for r in batches.values()), "total_rows": sum(sent.values())})

        observations = next_obs
        if pause:
            if stop is not None:
                stop.wait(pause)  # wakes up early when stopped
            else:
                time.sleep(pause)

    env.close()
    rounds = -(-steps // spr)  # rounds this pass used, rounded up
    for (host, table), n in sorted(sent.items()):
        print(f"published {n} rows to {host} {pub['dbms']}.{table} "
              f"(rounds {round_offset + 1}-{round_offset + rounds})")
    return start + datetime.timedelta(seconds=steps * env.seconds_per_time_step), round_offset + rounds