# 0.5412881800993953
import numpy as np



class Agent:

    def __init__(self, config=None):

        static = config["static"]

        action = config["spaces"]["action"]

        slots = static["action_slots"]

        edges = static["edges"]

        self.T = int(config["T"])

        self.slot_edges = np.asarray(slots["edge"], dtype=int)

        self.slot_k = np.asarray(slots["k"], dtype=int)

        self.slot_lanes = slots["lane"]

        self.capacity = np.asarray([edges["u0"][e] for e in self.slot_edges], dtype=float)

        self.tau0 = np.asarray(edges["tau0"], dtype=float)

        self.paths = []

        self.market_slots = {}

        demands = {(int(n), int(k)): d for d, (n, k) in enumerate(zip(static["sinks"]["node"], static["sinks"]["k"]))}

        for i, (e, k, lane) in enumerate(zip(self.slot_edges, self.slot_k, self.slot_lanes)):

            path = list(static["lanes"]["edges"][lane]) if lane is not None else [int(e)]

            self.paths.append(path)

            key = (int(edges["head"][path[-1]]), int(k))

            if key in demands:

                self.market_slots.setdefault(demands[key], []).append(i)

        self.sink_nodes = np.asarray(static["sinks"]["node"], dtype=int)

        self.sink_k = np.asarray(static["sinks"]["k"], dtype=int)

        self.override_qty = np.zeros(action["override_qty"]["shape"], dtype=float)

        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)



    def act(self, observation):

        o = observation

        allowed = np.asarray(o["action_mask"], dtype=float)

        flows = self.capacity.copy() * allowed

        week = int(o["week"][0])

        remaining = self.T - week + 1

        tau = np.asarray(o["graph_now.tau"], dtype=float)

        tau_observed = np.asarray(o["graph_now.tau.observed"], dtype=bool)

        for i, path in enumerate(self.paths):

            travel = float(np.sum(tau[path])) if np.all(tau_observed[path]) else float(np.sum(self.tau0[path]))

            if travel > remaining:

                flows[i] = 0.0



        forecast = np.asarray(o["demand_forecast.qty"], dtype=float)

        forecast_observed = np.asarray(o["demand_forecast.qty.observed"], dtype=bool)

        backlog = np.asarray(o["backlog.qty"], dtype=float)

        costs = np.asarray(o["graph_now.c"], dtype=float)

        for d, inds in self.market_slots.items():

            valid = forecast_observed[d]

            if not np.any(valid):

                continue

            lead = min(float(np.sum(tau[self.paths[i]])) if np.all(tau_observed[self.paths[i]]) else float(np.sum(self.tau0[self.paths[i]])) for i in inds)

            h = int(np.clip(round(lead) - 1, 0, forecast.shape[1] - 1))

            demand = max(0.0, float(backlog[d]) + float(forecast[d, h]))

            costs_i = np.maximum(0.0, costs[self.slot_edges[inds]])

            weights = self.capacity[inds] / np.power(1.0 + costs_i, 1.55) * allowed[inds]

            total = float(np.sum(weights))

            if total > 0:

                flows[inds] = np.minimum(self.capacity[inds], 1.35 * demand * weights / total)

            else:

                flows[inds] = 0.0

        return {"flows": flows * allowed, "override_qty": self.override_qty, "release_mode": self.release_mode}