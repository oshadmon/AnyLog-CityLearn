"""
The contract between the federated loop and one project.

The loop (client.py / server.py) only knows "run on each site's rows, combine centrally".
Everything project-specific lives in a Task: what to fetch, what local work means,
how results are combined, and how they are scored.

`state` is whatever the task needs it to be:
  - an MLP: a list of weight arrays, combined with weighted FedAvg
  - a tree model (e.g. OpenSTEF): per-site models, combined as an ensemble
  - a grid analysis (e.g. Power Grid Model): per-site results, combined by summing
The loop never looks inside it.

The GUI never looks inside it either: it shows what describe() and settings say, in plain
words for someone who may know neither AnyLog nor machine learning.
"""


class Task:
    name = "task"

    # Columns to fetch from AnyLog for this task (beyond what the UNS `where` selects on).
    columns: list = []

    # ---- how the GUI explains this task ----
    title = "Task"
    summary = ""                 # one sentence: what this task does
    local_step = ""              # what each site does with its own data
    combine_step = ""            # what the center does with what the sites send
    travels = ""                 # what actually moves between a site and the center
    metric = ""                  # key in evaluate() results that the GUI charts
    metric_label = ""            # e.g. "Average forecast error (kWh)"
    lower_is_better = True
    series = {}                  # state name -> label, e.g. {"federated": "Learned together"}

    # Knobs the GUI offers: [{"key", "label", "help", "min", "max", "step", "default"}]
    settings: list = []

    def describe(self) -> dict:
        keys = ["name", "title", "summary", "local_step", "combine_step", "travels",
                "metric", "metric_label", "lower_is_better", "series", "settings"]
        return {k: getattr(self, k) for k in keys}

    # ---- the work ----

    def init_state(self, seed: int):
        """Starting global state. Same seed -> same state everywhere."""
        raise NotImplementedError

    def train(self, state, rows: list):
        """Local work on one site's rows, starting from `state`. Returns the new state."""
        raise NotImplementedError

    def combine(self, results: list):
        """results = [(state, n_rows), ...] from every site -> the new global state."""
        raise NotImplementedError

    def evaluate(self, state, rows: list) -> dict:
        """Score `state` on one site's test rows. Returns a dict of metrics."""
        raise NotImplementedError

    def baseline(self, rows: list) -> dict:
        """Optional reference score that needs no training (e.g. naive forecast)."""
        return {}
