import numpy as np

class Agent:
    def __init__(self, config=None):
        static = config["static"]
        action = config["spaces"]["action"]
        slots = static["action_slots"]
        edges = static["edges"]
        self.slot_edges = np.asarray(slots["edge"], dtype=int)
        self.slot_k = np.asarray(slots["k"], dtype=int)
        self.capacity = np.asarray([edges["u0"][e] for e in self.slot_edges], dtype=float)
        self.tau0 = np.asarray(edges["tau0"], dtype=float)
        self.edge_head = np.asarray(edges["head"], dtype=int)
        self.sink_node = np.asarray(static["sinks"]["node"], dtype=int)
        self.sink_k = np.asarray(static["sinks"]["k"], dtype=int)
        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

    def act(self, observation):
        flows = self.capacity.copy()
        forecast = np.asarray(observation["demand_forecast.qty"], dtype=float)
        backlog = np.asarray(observation["backlog.qty"], dtype=float)
        allowed = np.asarray(observation["action_mask"], dtype=float)
        costs = np.asarray(observation["graph_now.c"], dtype=float)
        for d, (node, commodity) in enumerate(zip(self.sink_node, self.sink_k)):
            slots = [i for i, edge in enumerate(self.slot_edges)
                     if self.edge_head[edge] == node and self.slot_k[i] == commodity]
            if not slots:
                continue
            h = int(np.clip(np.min(self.tau0[self.slot_edges[slots]] - 1), 0,
                            forecast.shape[1] - 1))
            budget = 1.24 * max(0.0, backlog[d] + forecast[d, h])
            edge_cost = np.maximum(0.0, costs[self.slot_edges[slots]])
            weights = self.capacity[slots] / np.power(1.0 + edge_cost, 1.55) * allowed[slots]
            total = weights.sum()
            if total > 0:
                flows[slots] = np.minimum(self.capacity[slots], budget * weights / total)
            else:
                flows[slots] = 0.0
        flows *= allowed
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}