"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

import copy
import logging
import os

import numpy as np
import torch
from torch import nn

from platform_components.EdgeLake_functions.blockchain_EL_functions import fetch_data_from_db
from platform_components.lib.logger.logger_config import configure_logging
from platform_components.lib.modules.local_model_update import LocalModelUpdate
from platform_components.model_fusion_algorithms.FedAvg import FedAvg_aggregate

LOGICAL_DATABASE = os.getenv("LOGICAL_DATABASE")
TRAIN_TABLE = os.getenv("TRAIN_TABLE")
TEST_TABLE = os.getenv("TEST_TABLE")

# Columns the CityLearn publisher writes for every building and step.
RAW_COLUMNS = [
    "hour", "day_type",
    "outdoor_dry_bulb_temperature", "outdoor_relative_humidity",
    "diffuse_solar_irradiance", "direct_solar_irradiance",
    "carbon_intensity", "electricity_pricing",
    "non_shiftable_load", "solar_generation",
    "electrical_storage_soc", "net_electricity_consumption",
]

# Fixed scaling, identical on every node. Each node scaling by its own data would give
# the weights a different meaning per node, and averaging them would be meaningless.
# value -> (value - offset) / scale
SCALING = {
    "day_type": (4.0, 4.0),                     # 1..8
    "outdoor_dry_bulb_temperature": (20.0, 10.0),  # degC
    "outdoor_relative_humidity": (50.0, 50.0),  # %
    "diffuse_solar_irradiance": (0.0, 500.0),   # W/m2
    "direct_solar_irradiance": (0.0, 500.0),    # W/m2
    "carbon_intensity": (0.0, 0.5),             # kgCO2/kWh
    "electricity_pricing": (0.0, 0.5),          # $/kWh
    "non_shiftable_load": (0.0, 5.0),           # kWh
    "solar_generation": (0.0, 5.0),             # kWh
    "electrical_storage_soc": (0.0, 1.0),       # 0..1
    "net_electricity_consumption": (0.0, 5.0),  # kWh
}
LABEL_SCALE = 5.0  # label is next hour's net_electricity_consumption in kWh

# hour is encoded as sin / cos so 23:00 and 00:00 are neighbours
FEATURE_NAMES = ["hour_sin", "hour_cos"] + [c for c in RAW_COLUMNS if c != "hour"]
NUM_FEATURES = len(FEATURE_NAMES)


def _torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LoadForecastModel(nn.Module):
    """Small MLP: this hour's building state -> next hour's net consumption."""

    def __init__(self, num_features=NUM_FEATURES, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_features(row):
    """One raw row (dict) -> scaled feature vector. Missing values become 0 after scaling."""
    out = []
    hour = row.get("hour")
    hour = 0.0 if hour is None else float(hour)
    out.append(np.sin(2 * np.pi * hour / 24.0))
    out.append(np.cos(2 * np.pi * hour / 24.0))
    for name in RAW_COLUMNS:
        if name == "hour":
            continue
        value = row.get(name)
        offset, scale = SCALING[name]
        out.append(0.0 if value is None else (float(value) - offset) / scale)
    return out


class CityLearnDataHandler:
    """
    Federated next-hour load forecaster for CityLearn communities.

    Each training node reads only its own AnyLog operator (EXTERNAL_TCP_IP_PORT), so the
    building data never leaves the community. Round N trains on rows tagged round_number = N.
    """

    def __init__(self, node_name):
        configure_logging("node_server_data_handler")
        self.logger = logging.getLogger(__name__)
        self.node_name = node_name
        self.tcp_ip_port = os.getenv("EXTERNAL_TCP_IP_PORT")
        self.edgelake_node_url = f'http://{os.getenv("EXTERNAL_IP")}'
        self.db_name = LOGICAL_DATABASE
        self.device = _torch_device()
        # Same seed on every node, so all nodes start round 1 from identical weights.
        torch.manual_seed(int(os.getenv("MODEL_SEED", "0")))
        self.fl_model = self.model_def().to(self.device)
        self.logger.info("CityLearn load-forecast model on %s", self.device)

    # ---------- model / weights (the EdgeFL contract) ----------

    def model_def(self):
        return LoadForecastModel(hidden=int(os.getenv("HIDDEN_SIZE", "64")))

    def get_weights(self):
        return [param.detach().cpu().numpy() for param in self.fl_model.parameters()]

    def update_model(self, weights):
        if isinstance(weights, LocalModelUpdate):
            weights = weights.get("weights")
        params = list(self.fl_model.parameters())
        weights = list(weights)
        if len(weights) != len(params):
            raise ValueError(f"Expected {len(params)} parameter tensors, received {len(weights)}")
        with torch.no_grad():
            for param, value in zip(params, weights):
                tensor = torch.as_tensor(np.asarray(value), dtype=param.dtype, device=param.device)
                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(f"Shape mismatch for {tuple(param.shape)} vs {tuple(tensor.shape)}")
                param.copy_(tensor)

    def aggregate_model_weights(self, weights):
        return FedAvg_aggregate(weights)

    # ---------- data from AnyLog ----------

    @staticmethod
    def _normalize_row(row):
        """AnyLog may return lower-cased column names; map them back."""
        lower = {str(k).lower(): v for k, v in row.items()}
        return {name: lower.get(name) for name in RAW_COLUMNS + ["label", "building_id", "timestamp"]}

    def _query(self, table, where):
        columns = ", ".join(["building_id", "timestamp"] + RAW_COLUMNS + ["label"])
        return (f"sql {self.db_name} format=json and stat=false "
                f"SELECT {columns} FROM {table} WHERE {where}")

    def _fetch(self, table, where):
        if not self.tcp_ip_port:
            raise RuntimeError("EXTERNAL_TCP_IP_PORT is not set")
        query = self._query(table, where)
        self.logger.info("Querying %s", query)
        payload = fetch_data_from_db(self.edgelake_node_url, query, self.tcp_ip_port)
        if not isinstance(payload, dict) or not payload.get("Query"):
            raise IOError(f"No rows returned for: {query}. Response: {payload}")
        rows = [self._normalize_row(r) for r in payload["Query"]]
        rows = [r for r in rows if r["label"] is not None]
        rows.sort(key=lambda r: (str(r["building_id"]), str(r["timestamp"])))
        return rows

    @staticmethod
    def _to_arrays(rows):
        x = np.array([encode_features(r) for r in rows], dtype=np.float32)
        y = np.array([float(r["label"]) for r in rows], dtype=np.float32)
        # what the naive forecast says: next hour = this hour
        naive = np.array([float(r["net_electricity_consumption"] or 0.0) for r in rows], dtype=np.float32)
        return x, y, naive

    def _round_where(self, round_number, history_rounds=1):
        first = max(1, int(round_number) - max(1, int(history_rounds)) + 1)
        if first == int(round_number):
            return f"round_number = {int(round_number)}"
        return f"round_number >= {first} AND round_number <= {int(round_number)}"

    # ---------- training ----------

    def _fit(self, x_train, y_train):
        epochs = int(os.getenv("TRAIN_EPOCHS", "30"))
        batch_size = int(os.getenv("BATCH_SIZE", "64"))
        learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        features = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(y_train / LABEL_SCALE, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.fl_model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        best_loss, best_state = None, None

        for epoch in range(epochs):
            self.fl_model.train()
            order = torch.randperm(features.shape[0], device=self.device)
            total, batches = 0.0, 0
            for start in range(0, features.shape[0], batch_size):
                idx = order[start:start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.fl_model(features[idx]), labels[idx])
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            epoch_loss = total / max(batches, 1)
            self.logger.debug("Epoch %s/%s loss %.5f", epoch + 1, epochs, epoch_loss)
            if best_loss is None or epoch_loss < best_loss:
                best_loss, best_state = epoch_loss, copy.deepcopy(self.fl_model.state_dict())

        if best_state is not None:
            self.fl_model.load_state_dict(best_state)
        self.logger.info("Training done: %s samples, final loss %.5f", len(y_train), best_loss)
        return self.get_weights()

    def train(self, round_number):
        history = int(os.getenv("TRAIN_HISTORY_ROUNDS", "1"))
        rows = self._fetch(TRAIN_TABLE, self._round_where(round_number, history))
        x_train, y_train, _ = self._to_arrays(rows)
        self.logger.info("Round %s: training on %s rows (%s buildings)", round_number,
                         len(rows), len({r["building_id"] for r in rows}))
        return self._fit(x_train, y_train)

    # ---------- evaluation / inference ----------

    def _predict_kwh(self, x):
        self.fl_model.eval()
        with torch.no_grad():
            out = self.fl_model(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        return out.detach().cpu().numpy() * LABEL_SCALE

    @staticmethod
    def _metrics(y_true, y_pred):
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1e-9
        r2 = float(1 - np.sum(err ** 2) / ss_tot)
        return mae, rmse, r2

    def run_inference(self):
        """
        Score the current model on this community's held-out test rows and compare it with
        the naive forecast (next hour = this hour). INFERENCE_ROUND=all (default) uses every
        test round; a number uses that round only.
        """
        which = os.getenv("INFERENCE_ROUND", "all").strip().lower()
        where = "round_number >= 1" if which == "all" else f"round_number = {int(which)}"
        rows = self._fetch(TEST_TABLE, where)
        x_test, y_test, naive = self._to_arrays(rows)
        pred = self._predict_kwh(x_test)

        mae, rmse, r2 = self._metrics(y_test, pred)
        naive_mae, naive_rmse, _ = self._metrics(y_test, naive)
        skill = 1.0 - mae / naive_mae if naive_mae else 0.0
        sample = {i + 1: f"{rows[i]['building_id']} {rows[i]['timestamp']}: "
                         f"{pred[i]:.3f} --> {y_test[i]:.3f}" for i in range(min(10, len(rows)))}
        result = {
            "results": str(sample),
            "samples": int(len(y_test)),
            "buildings": len({r["building_id"] for r in rows}),
            "mae_kwh": round(mae, 4),
            "rmse_kwh": round(rmse, 4),
            "r2": round(r2, 4),
            "naive_mae_kwh": round(naive_mae, 4),
            "naive_rmse_kwh": round(naive_rmse, 4),
            # > 0 means the model beats "next hour = this hour"
            "skill_vs_naive": round(skill, 4),
        }
        self.logger.info("[Inference] %s", result)
        return result

    def direct_inference(self, data, labels=None):
        """
        Predict next hour's net consumption (kWh).
        `data` is one row or a list of rows. A row is either a dict with the RAW_COLUMNS
        names, or a list of 12 values in RAW_COLUMNS order:
        hour, day_type, outdoor_dry_bulb_temperature, outdoor_relative_humidity,
        diffuse_solar_irradiance, direct_solar_irradiance, carbon_intensity,
        electricity_pricing, non_shiftable_load, solar_generation,
        electrical_storage_soc, net_electricity_consumption
        """
        if isinstance(data, dict) or (data and not isinstance(data[0], (list, dict))):
            data = [data]
        rows = []
        for item in data:
            if isinstance(item, dict):
                rows.append({name: item.get(name) for name in RAW_COLUMNS})
            else:
                if len(item) != len(RAW_COLUMNS):
                    raise ValueError(f"Expected {len(RAW_COLUMNS)} values in this order: {RAW_COLUMNS}")
                rows.append(dict(zip(RAW_COLUMNS, item)))
        x = np.array([encode_features(r) for r in rows], dtype=np.float32)
        pred = self._predict_kwh(x)
        self.logger.info("[Inference] Edge inference complete")
        if labels:
            y = np.asarray(labels, dtype=np.float32).reshape(-1)
            mae, rmse, r2 = self._metrics(y, pred)
            return {"predictions": [round(float(p), 4) for p in pred],
                    "mae_kwh": round(mae, 4), "rmse_kwh": round(rmse, 4)}
        return [round(float(p), 4) for p in pred]
