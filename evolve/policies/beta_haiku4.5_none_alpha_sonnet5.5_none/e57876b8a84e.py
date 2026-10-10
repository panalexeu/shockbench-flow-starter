# 0.5451253201209032
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config["static"]
        self.n_over = len(st["override_slots"]["chokepoint"])
        self.n_rel = len(config["layout"]["release_pairs"])
        edges = st["edges"]
        self.tail = list(edges["tail"])
        self.head = list(edges["head"])
        self.c0 = np.array([0.0 if x is None else float(x) for x in edges["c0"]], dtype=float)
        self.u0 = np.array([np.inf if x is None else float(x) for x in edges["u0"]], dtype=float)
        self.slot_edge = list(st["action_slots"]["edge"])
        self.slot_k = list(st["action_slots"]["k"])
        self.slot_lane = list(st["action_slots"]["lane"])
        self.lane_edges = st["lanes"]["edges"]
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        self.sinks_pi = np.array([float(x) for x in st["sinks"]["pi"]], dtype=float)
        self.override_chokepoint = [int(x) for x in st["override_slots"]["chokepoint"]]
        self.override_k = [int(x) for x in st["override_slots"]["k"]]
        self.chokepoints = [int(x) for x in st["chokepoints"]] if "chokepoints" in st else []
        n = len(self.slot_edge)
        self.slot_last_head = []
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges) or not self.lane_edges[lane]:
                e = self.slot_edge[i]
            else:
                e = self.lane_edges[lane][-1]
            self.slot_last_head.append(int(self.head[e]))

    def act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        
        c = np.array(obs["graph_now.c"], dtype=float)
        tariff = np.array(obs["graph_now.tariff"], dtype=float)
        war_risk = np.array(obs["graph_now.war_risk"], dtype=int) if "graph_now.war_risk" in obs else np.zeros(len(self.chokepoints), dtype=int)
        chokepoint_open = np.array(obs["graph_now.open"], dtype=float) if "graph_now.open" in obs else np.ones(len(self.chokepoints), dtype=float)
        
        stock = np.clip(np.array(obs["stock.qty"], dtype=float), 0, None)
        backlog = np.clip(np.array(obs["backlog.qty"], dtype=float), 0, None)
        fc = np.clip(np.array(obs["demand_forecast.qty"], dtype=float), 0, None)
        mask = np.array(obs["action_mask"], dtype=float)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Slot capacity
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges):
                c_slot = cap[self.slot_edge[i]]
            else:
                le = self.lane_edges[lane]
                c_slot = min((cap[e] for e in le), default=0.0) if le else 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Track in-transit
        intransit = {}
        if "pipeline.qty" in obs:
            pq = np.array(obs["pipeline.qty"], dtype=float)
            pe = np.array(obs["pipeline.edge"], dtype=int)
            pk = np.array(obs["pipeline.k"], dtype=int)
            po = np.array(obs["pipeline.qty.observed"]) if "pipeline.qty.observed" in obs else np.ones_like(pq)
            for j in range(len(pq)):
                if po[j] > 0 and 0 <= pe[j] < len(self.head):
                    key = (int(self.head[pe[j]]), int(pk[j]))
                    if key in self.demand_idx:
                        intransit[key] = intransit.get(key, 0.0) + pq[j]
        
        slot_cost = np.zeros(n)
        for i in range(n):
            edge_idx = self.slot_edge[i]
            k = self.slot_k[i]
            slot_cost[i] = c[edge_idx] + tariff[edge_idx, k]
        
        # Demand pressure with longer horizon
        demand_need = {}
        for key, d in self.demand_idx.items():
            h = min(6, fc.shape[1])
            tot = backlog[d] * 3.0 + sum(fc[d, hh] * max(0, 1.0 - 0.12 * hh) for hh in range(h))
            tot -= intransit.get(key, 0.0)
            demand_need[key] = max(tot, 0.0)
        
        demand_pressure = np.zeros(n)
        penalty_priority = np.zeros(n)
        for i in range(n):
            key = (self.slot_last_head[i], int(self.slot_k[i]))
            demand_pressure[i] = demand_need.get(key, 0.0)
            d_idx = self.demand_idx.get(key, -1)
            if d_idx >= 0 and d_idx < len(self.sinks_pi):
                penalty_priority[i] = self.sinks_pi[d_idx]
            else:
                penalty_priority[i] = 1000.0
        
        # Allocate by source
        groups = {}
        for i in range(n):
            key = (int(self.tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        for key, idxs in groups.items():
            avail = stock[self.stock_idx[key]]
            idxs_sorted = sorted(idxs, key=lambda i: (-penalty_priority[i] * demand_pressure[i], slot_cost[i]))
            remaining = avail
            for i in idxs_sorted:
                slot_request = min(slot_cap[i], remaining)
                flows[i] = slot_request
                remaining -= slot_request
        
        flows = flows * mask
        
        # Tanker strategy: hold at high-risk or closed chokepoints
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        for j in range(self.n_rel):
            if j < len(self.override_chokepoint):
                chp = self.override_chokepoint[j]
                chp_idx = -1
                for cidx, c_node in enumerate(self.chokepoints):
                    if c_node == chp:
                        chp_idx = cidx
                        break
                if chp_idx >= 0:
                    risk = war_risk[chp_idx] if chp_idx < len(war_risk) else 0
                    openness = chokepoint_open[chp_idx] if chp_idx < len(chokepoint_open) else 1.0
                    if risk > 0 or openness < 0.5:
                        release_mode[j] = 2  # hold
        
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}