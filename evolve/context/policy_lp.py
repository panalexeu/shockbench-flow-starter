import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

COVER_WEEKS = 6.0  # stock target at each destination: lead time of its fastest route + this many weeks of flow


class Agent:
    """Each week: one linear program that ships, at least freight, what each destination lacks against its target."""

    def __init__(self, config=None):
        s, lay = config["static"], config["layout"]
        self.T = int(config["T"])
        e_ = s["edges"]
        self.head, self.tail = np.asarray(e_["head"], int), np.asarray(e_["tail"], int)
        self.tau0 = np.asarray(e_["tau0"], float)
        self.u0 = np.asarray([0.0 if x is None else x for x in e_["u0"]], float)
        self.c0 = np.asarray([0.0 if x is None else x for x in e_["c0"]], float)
        self.v = np.asarray(s["commodities"]["v"], float)
        self.lanes = s["lanes"]["edges"]
        self.lane_cps = s["lanes"]["chokepoints"]
        self.cp = {int(n): j for j, n in enumerate(lay["chokepoints"])}
        self.stock_idx = {tuple(x): j for j, x in enumerate(lay["stock_slots"])}
        self.sink_idx = {tuple(x): j for j, x in enumerate(lay["demands"])}
        self.pi = np.asarray(s["sinks"]["pi"], float)
        self.lot_keys = lay["lot_keys"]

        # each action slot: route (edges), source (node, k), destination (node, k), first edge
        sl = s["action_slots"]
        self.slots = []
        for e, k, lane in zip(sl["edge"], sl["k"], sl["lane"]):
            path = list(self.lanes[lane]) if lane is not None else [int(e)]
            src, dst = (int(self.tail[path[0]]), int(k)), (int(self.head[path[-1]]), int(k))
            self.slots.append((path, src, dst, int(e), lane))

        # nominal weekly flow into each destination, from the week-0 pipeline of the normal plan
        eid = {x: i for i, x in enumerate(e_["id"])}
        kid = {x: i for i, x in enumerate(s["commodities"]["id"])}
        lid = {x: i for i, x in enumerate(s["lanes"]["id"])}
        on_edge = {}
        for p in s["instance"]["initial_state"]["pipeline"]:
            key = (eid[p["edge"]], kid[p["k"]], lid.get(p["lane"]))
            on_edge[key] = on_edge.get(key, 0.0) + float(p["qty"])
        self.rate = {}
        for path, src, dst, e, lane in self.slots:
            q = on_edge.get((path[0], dst[1], lane), 0.0) / max(1.0, self.tau0[path[0]])
            self.rate[dst] = self.rate.get(dst, 0.0) + q

    def _dest_of(self, edge, lane):
        return int(self.head[self.lanes[lane][-1]]) if lane is not None and lane >= 0 else int(self.head[edge])

    def act(self, o):
        week = int(o["week"][0])
        u = np.where(o["graph_now.u.observed"] > 0, o["graph_now.u"], self.u0)
        c = np.where(o["graph_now.c.observed"] > 0, o["graph_now.c"], self.c0)
        tau = np.where(o["graph_now.tau.observed"] > 0, o["graph_now.tau"], self.tau0)
        tariff = o["graph_now.tariff"]
        opened = o["graph_now.open"]
        mask = o["action_mask"]

        # what each (node, k) holds, and what is already on its way to it
        stock = {key: max(0.0, float(o["stock.qty"][j])) for key, j in self.stock_idx.items()}
        avail = stock  # dispatch draws on last week's stock (supply is lifted after it)
        position = dict(stock)
        live = o["pipeline.qty.observed"] > 0
        for j in np.flatnonzero(live):
            lane = int(o["pipeline.lane"][j]) if o["pipeline.lane.observed"][j] else None
            key = (self._dest_of(int(o["pipeline.edge"][j]), lane), int(o["pipeline.k"][j]))
            position[key] = position.get(key, 0.0) + float(o["pipeline.qty"][j])
        for j, (n, k, lane, e) in enumerate(self.lot_keys):
            key = (self._dest_of(e, lane), int(k))
            position[key] = position.get(key, 0.0) + float(o["queue_lots.qty"][j].sum())
        for j in np.flatnonzero(o["wip.qty.observed"] > 0):
            key = (int(o["wip.node"][j]), int(o["wip.k"][j]))
            position[key] = position.get(key, 0.0) + float(o["wip.qty"][j])

        # usable routes this week: allowed, open, with capacity, arriving before the end
        usable, lead, cost, ub = [], {}, {}, {}
        for i, (path, src, dst, e, lane) in enumerate(self.slots):
            if not mask[i] or min(u[x] for x in path) <= 0:
                continue
            if lane is not None and any(opened[self.cp[n]] <= 0 for n in self.lane_cps[lane] if n in self.cp):
                continue
            t = float(sum(tau[x] for x in path))
            if week + t > self.T:
                continue
            usable.append(i)
            lead[i] = t
            cost[i] = sum(c[x] + tariff[x, dst[1]] * self.v[dst[1]] for x in path)
            ub[i] = min(u[x] for x in path)

        # target and deficit per destination
        dests = sorted({self.slots[i][2] for i in usable})
        deficit, penalty = {}, {}
        for d in dests:
            fastest = min(lead[i] for i in usable if self.slots[i][2] == d)
            if d in self.sink_idx:
                j = self.sink_idx[d]
                f = o["demand_forecast.qty"][j]
                target = float(f.mean()) * (fastest + COVER_WEEKS) + float(o["backlog.qty"][j])
                penalty[d] = self.pi[j]
            else:
                target = self.rate.get(d, 0.0) * (fastest + COVER_WEEKS)
                penalty[d] = self.v[d[1]]
            deficit[d] = max(0.0, target - position.get(d, 0.0))

        flows = np.zeros(len(self.slots))
        if not usable:
            return {"flows": flows}

        # LP variables: x_i (flow on each usable slot), then z_d (unmet deficit at each destination)
        nx, nz = len(usable), len(dests)
        col_d = {d: nx + j for j, d in enumerate(dests)}
        obj = np.concatenate([[cost[i] for i in usable], [penalty[d] for d in dests]])
        rows, cols, vals, b = [], [], [], []

        def row(entries, rhs):
            r = len(b)
            for col, val in entries:
                rows.append(r), cols.append(col), vals.append(val)
            b.append(rhs)

        # cover the deficit or pay for what is left:  -sum x_in - z_d <= -deficit_d
        for d in dests:
            row([(j, -1.0) for j, i in enumerate(usable) if self.slots[i][2] == d] + [(col_d[d], -1.0)], -deficit[d])
        # ship no more than the source holds
        for src in {self.slots[i][1] for i in usable}:
            row([(j, 1.0) for j, i in enumerate(usable) if self.slots[i][1] == src], avail.get(src, 0.0))
        # share each edge's capacity among the slots that start on it
        for e in {self.slots[i][3] for i in usable}:
            row([(j, 1.0) for j, i in enumerate(usable) if self.slots[i][3] == e], max(0.0, u[e]))

        A = coo_matrix((vals, (rows, cols)), shape=(len(b), nx + nz)).tocsr()
        bounds = [(0.0, ub[i]) for i in usable] + [(0.0, None)] * nz
        res = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if res.status == 0:
            flows[usable] = np.maximum(res.x[:nx], 0.0)
        return {"flows": flows}
