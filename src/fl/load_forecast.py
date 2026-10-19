"""
Next-hour load forecast: a small PyTorch MLP trained with weighted FedAvg.
Ported from citylearn_data_handler.py, without EdgeFL.

state = list of numpy weight arrays (one per model parameter).
"""
import numpy as np
import torch
from torch import nn

from .task import Task

# Columns the CityLearn publisher writes for every building and hour.
RAW_COLUMNS = [
    "hour", "day_type",
    "outdoor_dry_bulb_temperature", "outdoor_relative_humidity",
    "diffuse_solar_irradiance", "direct_solar_irradiance",
    "carbon_intensity", "electricity_pricing",
    "non_shiftable_load", "solar_generation",
    "electrical_storage_soc", "net_electricity_consumption",
]

# Fixed scaling, identical on every site. If each site scaled by its own data, the weights
# would mean different things per site and averaging them would be meaningless.
# value -> (value - offset) / scale
SCALING = {
    "day_type": (4.0, 4.0),                        # 1..8
    "outdoor_dry_bulb_temperature": (20.0, 10.0),  # degC
    "outdoor_relative_humidity": (50.0, 50.0),     # %
    "diffuse_solar_irradiance": (0.0, 500.0),      # W/m2
    "direct_solar_irradiance": (0.0, 500.0),       # W/m2
    "carbon_intensity": (0.0, 0.5),                # kgCO2/kWh
    "electricity_pricing": (0.0, 0.5),             # $/kWh
    "non_shiftable_load": (0.0, 5.0),              # kWh
    "solar_generation": (0.0, 5.0),                # kWh
    "electrical_storage_soc": (0.0, 1.0),          # 0..1
    "net_electricity_consumption": (0.0, 5.0),     # kWh
}
LABEL_SCALE = 5.0  # label = next hour's net_electricity_consumption, kWh

NUM_FEATURES = 2 + len(RAW_COLUMNS) - 1  # hour -> sin, cos


def encode(row):
    """One row (dict) -> scaled feature vector. Missing values become 0 after scaling."""
    hour = float(row.get("hour") or 0.0)
    out = [np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)]
    for name in RAW_COLUMNS[1:]:
        value = row.get(name)
        offset, scale = SCALING[name]
        out.append(0.0 if value is None else (float(value) - offset) / scale)
    return out


def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1e-9
    r2 = float(1 - np.sum(err ** 2) / ss_tot)
    return {"mae_kwh": round(mae, 4), "rmse_kwh": round(rmse, 4), "r2": round(r2, 4), "samples": int(len(y_true))}


class Model(nn.Module):
    """This hour's building state -> next hour's net consumption (scaled)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LoadForecastTask(Task):
    name = "load_forecast"
    columns = RAW_COLUMNS + ["label"]

    def __init__(self, epochs=30, batch_size=64, learning_rate=0.001, hidden=64, seed=0, **_):
        self.epochs, self.batch_size = int(epochs), int(batch_size)
        self.learning_rate, self.hidden, self.seed = float(learning_rate), int(hidden), int(seed)

    # ---------- weights <-> model ----------

    def _model(self, state):
        model = Model(self.hidden)
        with torch.no_grad():
            for param, value in zip(model.parameters(), state):
                param.copy_(torch.as_tensor(value))
        return model

    @staticmethod
    def _state(model):
        return [p.detach().cpu().numpy().copy() for p in model.parameters()]

    @staticmethod
    def _arrays(rows):
        rows = [r for r in rows if r.get("label") is not None]
        x = np.array([encode(r) for r in rows], dtype=np.float32)
        y = np.array([float(r["label"]) for r in rows], dtype=np.float32)
        return x, y, rows

    # ---------- the Task contract ----------

    def init_state(self, seed):
        torch.manual_seed(int(seed))
        return self._state(Model(self.hidden))

    def train(self, state, rows):
        x, y, _ = self._arrays(rows)
        if len(y) == 0:
            return state
        torch.manual_seed(self.seed)  # same shuffling on every run -> reproducible results
        model = self._model(state)
        features = torch.as_tensor(x)
        labels = torch.as_tensor(y / LABEL_SCALE)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(labels))
            for start in range(0, len(labels), self.batch_size):
                idx = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss_fn(model(features[idx]), labels[idx]).backward()
                optimizer.step()
        return self._state(model)

    def combine(self, results):
        """Weighted FedAvg: each site's weights count in proportion to its rows."""
        results = [(s, n) for s, n in results if n > 0]
        total = sum(n for _, n in results)
        return [sum(s[i] * (n / total) for s, n in results) for i in range(len(results[0][0]))]

    def predict(self, state, rows):
        """Next hour's net consumption in kWh, one value per row."""
        x = np.array([encode(r) for r in rows], dtype=np.float32)
        model = self._model(state).eval()
        with torch.no_grad():
            return model(torch.as_tensor(x)).numpy() * LABEL_SCALE

    def evaluate(self, state, rows):
        x, y, rows = self._arrays(rows)
        return metrics(y, self.predict(state, rows))

    def baseline(self, rows):
        """Naive forecast: next hour = this hour."""
        _, y, rows = self._arrays(rows)
        naive = np.array([float(r.get("net_electricity_consumption") or 0.0) for r in rows], dtype=np.float32)
        return metrics(y, naive)
