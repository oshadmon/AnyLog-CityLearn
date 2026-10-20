# AnyLog-CityLearn

**Project goal:** Demonstrate AnyLog / EdgeLake as the data layer underneath LF Energy projects.

**Description:** [CityLearn](https://www.citylearn.net/) generates the data. Its 17 buildings are split
into three communities, and each community's data is stored on its own AnyLog operator. Models then
run across the three communities without the raw data ever being collected in one place:

- **PyTorch, federated:** each community trains a next-hour load forecaster on its own data; only the
  models are combined (weighted FedAvg).
- **OpenSTEF:** LF Energy's short-term forecasting project; each community gets its own model, trained
  on its own history, forecasting its total load up to ~34 hours ahead.
- **Power Grid Model** (planned): grid analysis on the communities' aggregated load.

CityLearn is an open-source Gymnasium environment for building energy coordination and demand
response. It joined LF Energy as a project in September 2026. This demo uses the
`citylearn_challenge_2022_phase_all_plus_evs` dataset: 17 buildings with hourly weather, electricity
price, carbon intensity, load, solar generation, battery state and EV chargers, controlled by
CityLearn's business-as-usual agent.

The demo has three parts:

1. **Data.** Each community's hourly readings are published to its own AnyLog operator, and every
   building gets a name in AnyLog's Unified Namespace (UNS) that says where its data is.
2. **Forecasting.** Models find each community's data through the UNS and read it with AnyLog
   queries. The PyTorch model learns across communities with federated learning, and is scored
   against learning alone and a simple guess. OpenSTEF forecasts each community's load, and is
   scored against the same hour last week.
3. **GUI.** A guided, four-step web page for people who know neither AnyLog nor machine learning:
   where the data lives, how a building is found by name, training or forecasting, and what the
   results mean. Buttons load the history, train, and stream live data.

The models share the same data access: replacing or adding one (another LF Energy project) doesn't
change how data is published, found or queried.

**Architecture:** Three AnyLog operators, one per community. The master and query roles run either
on the operators themselves or on separate nodes.

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
               AnyLog queries, routed to each operator
                 +------------+-------------+
                 v                          v
       PyTorch, federated            OpenSTEF, one model
       (models combined)             per community
                 +------------+-------------+
                              |
                         browser GUI
```

## Requirements

- An AnyLog network with one operator per community (3 here), and REST access to at least one node
- Python 3.11 or 3.12
- Two virtual environments, because CityLearn needs numpy < 2 and OpenSTEF needs numpy 2:
  - `venv`: `requirements.txt` (CityLearn, PyTorch, the GUI). PyTorch is a large download.
  - `venv-openstef`: `requirements-openstef.txt` (only needed for OpenSTEF)

## Steps

1. Deploy a small AnyLog network with 3 operators.
2. Install:
   ```shell
   python -m venv venv
   venv\Scripts\activate                # Windows; source venv/bin/activate on Linux / macOS
   pip install -r requirements.txt

   # only for OpenSTEF; the GUI finds it here on its own
   python -m venv venv-openstef
   venv-openstef\Scripts\pip install -r requirements-openstef.txt
   ```
3. Update the [configurations](./configs/configs.yaml). Key settings: the operators, the query node,
   the logical database name, and `training_type` (`fl` or `openstef`).

   | Section | What it sets |
   |---|---|
   | `operators` | One entry per community: the operator's REST `host` (ip:port) and its `community_id`. Buildings are split across operators in order (17 over 3 gives 6 / 6 / 5). |
   | `query` | Optional. The node used for queries. Defaults to the first operator. |
   | `anylog` | How every REST call is made: user agent, timeout, publishing topic, mapping policy file. |
   | `simulation` | CityLearn dataset (`schema`), hours to simulate (`steps`: 168 is a week, 8760 a year), `seed`, start date. |
   | `publish` | Logical database (`dbms`), base table name, and the pause between simulated hours for live data. |
   | `training_type` | What step 3 of the GUI (and `main.py --train`) runs: `fl` (PyTorch, federated) or `openstef`. |
   | `fl` | Training fields added at publish time: the label (next hour's consumption), hours per round (168 = one week), and the share of each round held back for testing. |
   | `training` | Optional. Federated settings: task, UNS root, tables, rounds, epochs, learning rate, model size, seed, results file. |
   | `openstef` | OpenSTEF settings: model type, quantiles, history, the hourly aggregates, results file, and `python` (the OpenSTEF venv's python; null = `venv-openstef` in the repo). |
   | `gui` | Optional. `port` for the GUI. |

4. Start the application (in `venv`):
   ```shell
   python ./src/gui.py
   ```
   Step 1's **Load history** sets up the operators and publishes the simulation. **Start live data**
   keeps publishing new weeks. Step 3 trains (or forecasts) on what's in AnyLog.

The same actions run from the command line with `src/main.py`; see `python src/main.py --help`.

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
Weeks with less than 14 days of history before them are skipped, since OpenSTEF's lag features
reach back 14 days. The weather in the window is the measured weather, standing in for the weather
forecast an operator would use.

**In the GUI** (`training_type: openstef`), the GUI stays in `venv` and runs OpenSTEF with the
OpenSTEF venv's python. Step 3 shows each week's forecast against what really happened, the error
week by week, plain-language explanations, and a "For specialists" section with every score and the
exact AnyLog query.

**From the command line** (in `venv-openstef`):

```
python src/main.py --skip-insert --train --training-type openstef            # every community, every week
python src/main.py --skip-insert --train --training-type openstef --community community_a --rounds 3 8
python src/main.py --skip-insert --train --training-type openstef --resume   # only weeks not yet forecast
python src/main.py --skip-insert --train --training-type openstef --follow   # keep forecasting new live weeks
```

`--follow` forecasts each new week as live data completes it, while the live data itself is
published from `venv` (the GUI's "Start live data", or `main.py --continuous-insert`). Results
(scores and every forecast, with its quantiles) go to `results/openstef_results.json`.

## Adding another LF Energy project

**A model that learns across communities** implements `src/fl/task.py` and plugs into the
federated loop:

| Method | What it does |
|---|---|
| `init_state(seed)` | Starting state, identical on every site |
| `train(state, rows)` | Local work on one site's rows |
| `combine([(state, n_rows), ...])` | Combine the sites' results into one |
| `evaluate(state, rows)` | Score on one site's test rows |
| `baseline(rows)` | Optional reference score that needs no training |

The class also carries the plain-language text and settings the GUI shows, so the GUI needs no
changes. Register it in `TASKS` in `src/fl/server.py` and set `training.task` in the config.

**A project that runs per community** (like OpenSTEF) or on aggregates (like Power Grid Model)
follows `src/fl_openstef/` instead: get the community's objects from the UNS, query what it needs
through `src/rest.py`, and report progress as events for the GUI.