"""
One client per site (community). It only ever queries its own site's UNS leaves.

Narrow interface on purpose: train(round, states) and evaluate(states). Today the server
calls these in-process; later the same two calls can sit behind a small HTTP service next
to each operator, so only states (weights) travel and the server code does not change.
"""
from .data import fetch


def round_where(round_number, history_rounds=1):
    """Train on this round only (history_rounds=1) or on the last N rounds."""
    last = int(round_number)
    first = max(1, last - max(1, int(history_rounds)) + 1)
    if first == last:
        return f"round_number = {last}"
    return f"round_number >= {first} AND round_number <= {last}"


class Client:
    def __init__(self, site, train_leaves, test_leaves, task, conn, history_rounds=1):
        self.site, self.task, self.conn = site, task, conn
        self.train_leaves, self.test_leaves = train_leaves, test_leaves
        self.history_rounds = history_rounds
        self._test_rows = None

    @property
    def assets(self):
        return [leaf.name for leaf in self.train_leaves]

    def last_round(self):
        rows = fetch(self.conn, self.train_leaves, ["MAX(round_number) as last_round"])
        return int(rows[0]["last_round"]) if rows and rows[0].get("last_round") is not None else 0

    def train(self, round_number, states):
        """states = {"federated": s, "local": s, ...}. Each is trained on the same rows.
        Returns (new states, rows used)."""
        rows = fetch(self.conn, self.train_leaves, self.task.columns,
                     round_where(round_number, self.history_rounds))
        return {name: self.task.train(state, rows) for name, state in states.items()}, len(rows)

    def evaluate(self, states):
        """Score every state, plus the task's baseline, on this site's test rows (all rounds)."""
        if self._test_rows is None:  # fixed test set: fetched once, so rounds stay comparable
            self._test_rows = fetch(self.conn, self.test_leaves, self.task.columns)
        scores = {name: self.task.evaluate(state, self._test_rows) for name, state in states.items()}
        baseline = self.task.baseline(self._test_rows)
        if baseline:
            scores["baseline"] = baseline
        return scores