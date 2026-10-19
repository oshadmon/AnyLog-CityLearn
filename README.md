# AnyLog-CityLearn

**Project Goal**: Demonstrate AnyLog / EdgeLake as the underlying data layer for LF Energy projects. 

**Description**: With [CityLearn](https://www.citylearn.net/) as the data generating layer of the project, 
demonstrate using different types of  modeling tools, torch, _OpenSTEF_ or _Power Grid Model_, to execute Federated Learning 
across the 3 different communities within CityLearn; with data being stored on different AnyLog operator nodes.

CityLearn is an open-source Gymnasium environment for building energy coordination and demand response. It joined 
LF Energy as a project in September 2026.

This demo uses the `citylearn_challenge_2022_phase_all_plus_evs` dataset: 17 buildings with hourly weather, electricity 
price, carbon intensity, load, solar generation, battery state and EV chargers. The buildings are controlled by 
CityLearn's business-as-usual agent.

The demo has three parts:

1. **Data.** A CityLearn simulation of 17 buildings, split into three communities. Each
   community's hourly readings are published to its own AnyLog operator.
2. **Federated learning.** Each community trains a next-hour load forecaster on its own
   data. Only the models are combined (weighted FedAvg), and the result is scored against
   learning alone and against a simple guess.
3. **GUI.** A guided, four-step web page: where the data lives, how a building is found
   by name, training round by round, and what the results mean.

The forecaster is a stand-in. The training loop is written so that LF Energy tools such as
OpenSTEF or Power Grid Model can replace it without changing the loop or the GUI.


**Architecture**: Deployed are 3 AnyLog operators, with Master and Query either residing as part of the 3 operators or
independently of the 3 operators. 

Each site's energy data stays on its own AnyLog operator. Analysis runs across all sites without collecting the raw data 
in one place, and a browser GUI shows the whole thing to people who know neither AnyLog nor machine learning.

```
             CityLearn simulation (17 buildings, hourly)
                              |
                 publish (REST, one topic per row)
          +-------------------+-------------------+
          v                   v                   v
   AnyLog operator     AnyLog operator     AnyLog operator
    community_a          community_b         community_c
    (6 buildings)        (6 buildings)       (5 buildings)
          |                   |                   |
          +------ UNS: one name per building -----+
                              |
             federated learning: each community trains
             on its own rows, only the models are combined
                              |
                        browser GUI
```

## Requirements

- An AnyLog network with 3 operators (each on its own cluster) 
- Python 3.11 or 3.12
- A running AnyLog network with one operator per community, and REST access to at least
  one node
- `pip install -r requirements.txt` (PyTorch is a large download)

## Setup

```
git clone <this repo>
cd AnyLog-CityLearn
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Configure

Everything lives in `configs/configs.yaml`.

| Section | What it sets |
|---|---|
| `operators` | One entry per community: the operator's REST `host` (ip:port) and its `community_id`. Buildings are split across operators in order (17 over 3 gives 6 / 6 / 5). |
| `query` | Optional. The node used for queries. Defaults to the first operator. |
| `simulation` | CityLearn dataset (`schema`), hours to simulate (`steps`: 168 is a week, 8760 a year), `seed`, start date. |
| `publish` | Logical database (`dbms`), base table name, and a pause between simulated hours for live streaming. |
| `fl` | Training fields added at publish time: the label (next hour's consumption), hours per round (168 = one week), and the share of each round held back for testing. |
| `training` | Optional. Task name, UNS root, tables, rounds, epochs, learning rate, model size, seed, results file. Sensible defaults apply when it's missing. |
| `gui` | Optional. `port` for the GUI (default 8080). |

Example `training` and `gui` sections:

```yaml
training:
  task: load_forecast
  uns_root: citylearn_demo
  train_table: building_obs_train
  test_table: building_obs_test
  rounds: null            # null = every round all communities have
  history_rounds: 1       # 1 = round N trains on week N only
  epochs: 30
  batch_size: 64
  learning_rate: 0.001
  hidden: 64
  seed: 0
  results: results/fl_results.json

gui:
  port: 8080
```

## Run

All commands run from the repo root, with the venv active.

**Publish the data** (simulate CityLearn and send it to AnyLog, then print row counts per node):

```
python src/main.py
```

Publishing again adds the rows a second time. To start over, drop the tables and the
UNS / object policies on the operators first.

**Train from the command line** on data already in AnyLog:

```
python src/main.py --skip-insert --train
```

Each round prints the forecast error (MAE, kWh) per community for three approaches:
learned together, learned alone, and the simple guess (next hour = this hour). Results are
written to `results/fl_results.json` after every round.

**Start the GUI:**

```
python src/gui.py
```

It opens `http://localhost:8080` in the browser. The GUI has three buttons, one for each action:

| Button | Where | What it does |
|---|---|---|
| **Load history** | Step 1 | Sets up the mapping policy and message client on every operator, then publishes the simulated history once. Pressing it again adds the same weeks a second time. |
| **Start training** | Step 3 | Federated training on what's in AnyLog. Stop it at any point. After the first run, tick "Continue from what was learned before" to train only on newer weeks, with a from-scratch model alongside for comparison. |
| **Start live data** | Step 1 | Keeps publishing new hours after the last week in AnyLog, timestamped from now, until stopped. Train again to learn from the new weeks; the week still being filled is left out. |

The command line runs the same actions (`src/pipeline.py` holds the shared code):

```
python src/main.py --skip-insert --train --resume              # only the weeks since the saved model
python src/main.py --skip-insert --train --continuous-insert   # live data + training on each new week
```

The trained model is saved to `results/fl_model.pkl` after every round.

## How the data is organized in AnyLog

**Rows.** One row per building per hour: `timestamp`, `community_id`, `building_id`,
`time_step`, weather, price, carbon intensity, `non_shiftable_load`, `solar_generation`,
`electrical_storage_soc`, `net_electricity_consumption`, the controller's action and reward.
Three fields are added for training:

- `label`: the same building's `net_electricity_consumption` one hour later
- `round_number`: one simulated week per round
- `data_type`: `train` for the first 80% of each week, `test` for the last 20% (split by
  time, not at random)

Rows go to `mydb.building_obs_train` and `mydb.building_obs_test`. With a 168-hour week,
each building has 135 training rows and 33 test rows per round.

**Topics and the UNS.** Each row is published to the operator that owns its community, on
the topic `citylearn-demo/<table>/<community>/<building>`. The AnyLog message client turns
the topic into a Unified Namespace (UNS) path:

```
citylearn_demo/building_obs_train/community_a/building_1
```

Each building's UNS entry points to an object that says where its rows are:
`{"dbms": "mydb", "table": "building_obs_train", "where": "building_id = 'Building_1'"}`.

**Queries.** Every call to AnyLog, over REST, is in one file: `src/rest.py`. How the calls are made (user agent, timeout, topic, mapping policy file) comes from the `anylog` section of the config:

- `blockchain get uns bring.extend object` lists every building with its object.
- `sql mydb format=json and stat=false "SELECT ..."` with the header
  `destination: network` fetches a community's rows. The query node passes it to the
  operators holding the table and returns the combined answer.

No code needs to know which node holds which building; the UNS supplies it.

## Federated learning

Each round:

1. Every community trains two models on that week's rows: the shared model it received
   from the center, and its own local-only model (never shared).
2. The center combines the shared models into one with weighted FedAvg (weighted by rows).
3. Every model is scored on each community's held-back test rows from all weeks, together
   with the simple guess.

Feature scaling is fixed and identical everywhere (never per community), so averaged
weights mean the same thing on every site. The hour of day is encoded as sin / cos.

With only a few weeks of data, learning alone often wins, because a community's own
patterns are easy to learn and the shared model is a compromise between neighborhoods.
Publish a full simulated year (`steps: 8760`) before drawing conclusions.

## OpenSTEF forecasts

[OpenSTEF](https://lfenergy.org/projects/openstef/) is LF Energy's short-term energy forecasting
project. `src/fl_openstef/` runs it on the data in AnyLog: each community gets its own OpenSTEF
model, trained where its data lives; nothing is combined.

For every community and every week's held-back window (the last ~20% of the week):

1. The UNS gives the community's objects: dbms, tables, and the WHERE for each building.
2. One query returns the community's hourly history up to the window, across both tables
   (`include=(building_obs_test)`), bucketed with `increments(hour, 1, timestamp)` and tagged
   with `@table_name`. OpenSTEF trains on it.
3. A second query returns the last 14 days plus the window; OpenSTEF forecasts the window
   (up to ~34 hours ahead) with P10/P50/P90 quantiles.
4. The forecast is scored against the actual load, next to a simple guess: the same hour last week.

The query per hour: total load (`SUM(net_electricity_consumption)`), average temperature,
humidity, solar radiation and price, and the number of buildings that reported. Hours where that
count isn't the community's building count (still being filled, or loaded twice) are dropped.

OpenSTEF runs in its own venv, because it needs numpy 2 and CityLearn needs numpy below 2:

```
python -m venv venv-openstef
venv-openstef\Scripts\activate
pip install -r requirements-openstef.txt
python src/main.py --skip-insert --train --training-type openstef            # every community, every week
python src/main.py --skip-insert --train --training-type openstef --community community_a --rounds 3 8
python src/main.py --skip-insert --train --training-type openstef --resume   # only weeks not yet forecast
python src/main.py --skip-insert --train --training-type openstef --follow   # keep forecasting new live weeks
```

`--follow` forecasts each new week as live data completes it, while the live data itself is
published from the other venv (the GUI's "Start live data", or `main.py --continuous-insert`).
Set `training_type: openstef` in the config to make it the default.

Settings are in the `openstef:` section of `configs/configs.yaml`; results (scores and every
forecast, with its quantiles) go to `results/openstef_results.json`. Weeks with less than 14 days
of history before them are skipped, since OpenSTEF's lag features reach back 14 days. The weather
in the window is the measured weather, standing in for the weather forecast an operator would use.

## Adding another LF Energy project

Everything project-specific lives in one class that implements `src/fl/task.py`:

| Method | What it does |
|---|---|
| `init_state(seed)` | Starting state, identical on every site |
| `train(state, rows)` | Local work on one site's rows |
| `combine([(state, n_rows), ...])` | Combine the sites' results into one |
| `evaluate(state, rows)` | Score on one site's test rows |
| `baseline(rows)` | Optional reference score that needs no training |

`state` is whatever the project needs: model weights, per-site tree models (OpenSTEF,
where FedAvg doesn't apply), or per-site aggregates for grid analysis (Power Grid Model,
PowSyBl). The class also carries the plain-language text and settings the GUI shows, so the
GUI needs no changes. Register the class in `TASKS` in `src/fl/server.py` and set
`training.task` in the config.

## Project layout

```
AnyLog-CityLearn/
├── configs/
│   ├── configs.yaml            all settings
│   └── mapping_policy.json     the AnyLog mapping policy (id and dbms filled in at run time)
├── results/                    training results (created on first run)
├── requirements.txt
└── src/
    ├── main.py                 publish, check, train (command line)
    ├── pipeline.py             the publishing actions shared by main.py and the GUI
    ├── gui.py                  start the web GUI
    ├── data_gen_citylearn.py   CityLearn simulation -> rows
    ├── rest.py                 every AnyLog call: publishing, UNS, row and hourly queries
    ├── support.py              config helpers
    ├── fl_openstef/            OpenSTEF on AnyLog data (own venv)
    │   ├── forecast.py         query rows -> OpenSTEF train / forecast
    │   ├── backtest.py         week by week, per community
    ├── fl/
    │   ├── task.py             the contract every project implements
    │   ├── load_forecast.py    the PyTorch next-hour forecaster
    │   ├── client.py           one per community
    │   └── server.py           the rounds, combining, scoring
    └── web/
        ├── app.py              GUI backend (FastAPI)
        └── static/index.html   the page
```

## Known limitations

- **Trainers run on one computer.** Each community's trainer reads only its own data
  through AnyLog, and only models are shared between communities, but all three trainers
  run in one Python process. In production, each runs next to its own operator. The client
  interface (`train` / `evaluate`) is kept narrow so it can move behind a small service per
  operator without changing the server.
- **Publishing sends one HTTP request per row.** Fine for weeks of data; slow for a full
  year.
- **Re-publishing duplicates rows.** There is no clean-up step yet.

## Related

- [CityLearn](https://www.citylearn.net/)
- [LF Energy projects](https://lfenergy.org/projects/)
- [EdgeFL](https://github.com/AnyLog-co/EdgeFL), the federated-learning reference this
  demo's training loop is modeled on (not a dependency)