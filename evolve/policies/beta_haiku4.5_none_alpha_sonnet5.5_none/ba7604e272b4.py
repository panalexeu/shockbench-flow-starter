# 0.5501772552587368
import numpy as np

class Agent:
    SINK_CAP = False
    MARGIN = 1.5
    H = 8

    def __init__(self, config=None):
        st = config["static"]
        self.n_over = len(st["override_slots"]["chokepoint"])
        self.n_rel = len(config["layout"]["release_pairs"])
        edges = st["edges"]
        self.u0 = np.array([np.inf if x is None else float(x) for x in edges["u0"]], dtype=float)
        self.slot_edge = [int(x) for x in st["action_slots"]["edge"]]
        self.slot_k = [int(x) for x in st["action_slots"]["k"]]
        self.slot_lane = list(st["action_slots"]["lane"])
        self.lane_edges = st["lanes"]["edges"]
        self.lane_chk = st["lanes"]["chokepoints"]
        self.edge_head = [int(x) for x in edges["head"]]
        self.edge_tail = [int(x) for x in edges["tail"]]
        self.chk_nodes = [int(x) for x in config["layout"]["chokepoints"]]
        self.chk_pos = {n: i for i, n in enumerate(self.chk_nodes)}
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        n = len(self.slot_edge)
        self.slot_head = []
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges) or not self.lane_edges[lane]:
                self.slot_head.append(self.edge_head[self.slot_edge[i]])
            else:
                self.slot_head.append(self.edge_head[int(self.lane_edges[lane][-1])])

    def _fallback(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        cap = np.where(np.isfinite(u), u, 0.0)
        cap = np.clip(cap, 0, None)
        n = len(self.slot_edge)
        flows = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None:
                flows[i] = cap[self.slot_edge[i]]
            else:
                flows[i] = min(cap[int(e)] for e in self.lane_edges[lane])
        flows = flows * np.array(obs["action_mask"], dtype=float)
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}

    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            return self._fallback(obs)

    def _act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        c = np.array(obs["graph_now.c"], dtype=float)
        tariff = np.array(obs["graph_now.tariff"], dtype=float)
        opn = np.array(obs["graph_now.open"], dtype=float)
        stock = np.clip(np.array(obs["stock.qty"], dtype=float), 0, None)
        backlog = np.clip(np.array(obs["backlog.qty"], dtype=float), 0, None)
        fc = np.clip(np.array(obs["demand_forecast.qty"], dtype=float), 0, None)
        mask = np.array(obs["action_mask"], dtype=float)
        n = len(self.slot_edge)

        intransit = {}
        pq = np.array(obs["pipeline.qty"], dtype=float)
        pe = np.array(obs["pipeline.edge"], dtype=int)
        pk = np.array(obs["pipeline.k"], dtype=int)
        po = np.array(obs["pipeline.qty.observed"])
        for j in range(len(pq)):
            if po[j] > 0 and 0 <= pe[j] < len(self.edge_head):
                key = (self.edge_head[pe[j]], int(pk[j]))
                intransit[key] = intransit.get(key, 0.0) + pq[j]

        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges) or not self.lane_edges[lane]:
                cs = cap[self.slot_edge[i]]
            else:
                cs = min(cap[int(e)] for e in self.lane_edges[lane])
                for ch in self.lane_chk[lane]:
                    p = self.chk_pos.get(int(ch))
                    if p is not None and p < len(opn):
                        if opn[p] < 0.05:
                            cs = 0.0
            slot_cap[i] = max(cs, 0.0)

        need = {}
        press = np.zeros(n)
        for i in range(n):
            key = (self.slot_head[i], self.slot_k[i])
            if key in self.demand_idx:
                d = self.demand_idx[key]
                p = backlog[d] * 2.5
                tot = backlog[d]
                for h in range(min(self.H, fc.shape[1])):
                    p += fc[d, h] * np.exp(-0.1 * h)
                    tot += fc[d, h]
                press[i] = p
                have = intransit.get(key, 0.0)
                if key in self.stock_idx:
                    have += stock[self.stock_idx[key]]
                need[key] = max(tot * self.MARGIN - have, 0.0)

        slot_cost = np.array([max(c[self.slot_edge[i]] + tariff[self.slot_edge[i], self.slot_k[i]], 0.0) for i in range(n)])

        groups = {}
        for i in range(n):
            key = (self.edge_tail[self.slot_edge[i]], self.slot_k[i])
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)

        flows = np.zeros(n)
        for key, idxs in groups.items():
            rem = stock[self.stock_idx[key]]
            for i in sorted(idxs, key=lambda i: (-press[i], slot_cost[i])):
                if mask[i] <= 0:
                    continue
                q = min(slot_cap[i], rem)
                sk = (self.slot_head[i], self.slot_k[i])
                if self.SINK_CAP and sk in need:
                    q = min(q, need[sk])
                if q > 0:
                    flows[i] = q
                    rem -= q
                    if self.SINK_CAP and sk in need:
                        need[sk] -= q
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}
