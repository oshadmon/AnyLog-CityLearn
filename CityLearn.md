# CityLearn Demo

[CityLearn](https://www.citylearn.net/) is an LF Energy project: a simulation environment for
building energy and grid-interactive communities. In this demo, CityLearn's buildings are split
into communities, each community's data is published into its own AnyLog / EdgeLake operator,
and EdgeFL trains a shared **next-hour electricity forecaster** across the communities. No
building data leaves its operator; only model weights move.

The data handler is `edgefl/platform_components/data_handlers/citylearn_data_handler.py`
(PyTorch). This is a starter model meant to prove the pipeline; it can later be swapped for an
LF Energy forecasting project such as OpenSTEF.

# Demo execution
The demo assumes 3 operators (one per community) and 1 aggregator.

| Training node | Community | AnyLog node | REST | TCP |
|---|---|---|---|---|
| citylearn1 (8081) | community_a (6 buildings) | anylog-standalone-operator | 32049 | 32048 |
| citylearn2 (8082) | community_b (6 buildings) | anylog-operator1 | 32149 | 32148 |
| citylearn3 (8083) | community_c (5 buildings) | anylog-operator2 | 32159 | 32158 |

## Insert data
Run the CityLearn publisher with `fl.enabled: true`. Every row gets:
- `label`: the same building's `net_electricity_consumption` one hour later
- `round_number`: one simulated week per round (`fl.steps_per_round: 168`)
- `data_type`: `train` for the first 80% of each week, `test` for the last 20%

and is written to `building_obs_train` or `building_obs_test` on the community's operator.

## Validate data
```bash
run client () sql mydb format=table and include=(building_obs_test) and extend=(+node_name, @table_name) "select round_number, count(*) from building_obs_train group by round_number"
```
For a 4-week run (`simulation.steps: 672`), each operator should show rounds 1-4 in both tables
(e.g. 810 train / 198 test rows per round for a 6-building community).

## Define env files
Update the files in `edgefl/env_files/citylearn/`:
- `GITHUB_DIR` --> the path to the EdgeFL repo
- `EXTERNAL_IP` / `EXTERNAL_TCP_IP_PORT` --> each node's REST / TCP address (`get connections`)
- `EDGELAKE_DOCKER_CONTAINER_NAME` --> the container running that AnyLog node
  (set `EDGELAKE_DOCKER_RUNNING="False"` if the nodes are not containers)

Keep `MODEL_SEED` and `HIDDEN_SIZE` identical on every node so all nodes start from the same model.

## Start the aggregator and training nodes
```bash
cd edgefl
dotenv -f env_files/citylearn/citylearn-agg.env run -- uvicorn platform_components.aggregator.aggregator_server:app --host 0.0.0.0 --port 8080
dotenv -f env_files/citylearn/citylearn1.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8081
dotenv -f env_files/citylearn/citylearn2.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8082
dotenv -f env_files/citylearn/citylearn3.env run -- uvicorn platform_components.node.node_server:app --host 0.0.0.0 --port 8083
```

## Initialize and start training
```bash
curl -X POST http://localhost:8080/init \
-H "Content-Type: application/json" \
-d '{
  "nodeUrls": ["http://localhost:8081", "http://localhost:8082", "http://localhost:8083"],
  "index": "citylearn-fl"
}'

curl -X POST http://localhost:8080/start-training \
-H "Content-Type: application/json" \
-d '{
  "totalRounds": 4,
  "minParams": 3,
  "index": "citylearn-fl"
}'
```
`totalRounds` must not exceed the number of rounds published (4 for a 4-week run).

## Inference
Each node scores the current model on its own community's test rows and compares it with the
naive forecast "next hour = this hour":
```bash
curl -X POST http://localhost:8081/inference/citylearn-fl
curl -X POST http://localhost:8082/inference/citylearn-fl
curl -X POST http://localhost:8083/inference/citylearn-fl
```
Key fields: `mae_kwh`, `naive_mae_kwh`, and `skill_vs_naive` (above 0 means the model beats the
naive forecast).

Direct inference on one building-hour (values in this order: hour, day_type,
outdoor_dry_bulb_temperature, outdoor_relative_humidity, diffuse_solar_irradiance,
direct_solar_irradiance, carbon_intensity, electricity_pricing, non_shiftable_load,
solar_generation, electrical_storage_soc, net_electricity_consumption):
```bash
curl -X POST http://localhost:8081/infer \
-H "Content-Type: application/json" \
-d '{
  "input": [18, 1, 24.4, 60, 120, 300, 0.25, 0.22, 1.8, 0.6, 0.4, 1.2],
  "index": "citylearn-fl"
}'
```
The output is next hour's predicted net consumption in kWh.
