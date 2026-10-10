# -999
# error: 832 of 832 weeks crashed, first: episode 0, week 0: NameError: name 'j' is not defined
import numpy as np

class Agent:
    def __init__(self, config=None):
        st = config["static"]
        self.st = st
        self.config = config
        self.n_over = len(st["override_slots"]["chokepoint"])
        self.n_rel = len(config["layout"]["release_pairs"])
        edges = st["edges"]
        self.u0 = np.array([np.inf if x is None else float(x) for x in edges["u0"]], dtype=float)
        self.c0 = np.array([float(x) if x is not None else 0.0 for x in edges["c0"]], dtype=float)
        self.slot_edge = list(st["action_slots"]["edge"])
        self.slot_k = list(st["action_slots"]["k"])
        self.slot_lane = list(st["action_slots"]["lane"])
        self.lane_edges = st["lanes"]["edges"]
        self.edge_head = list(edges["head"])
        self.edge_tail = list(edges["tail"])
        
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        
        # Override slot info
        self.override_chokepoint = [int(x) for x in st["override_slots"]["chokepoint"]]
        self.override_k = [int(x) for x in st["override_slots"]["k"]]
        self.override_out_edge = [int(x) for x in st["override_slots"]["out_edge"]]
        
        # Release pair info
        self.release_chokepoint = [int(x) for x in config["layout"]["release_pairs"][j][0] if j < len(config["layout"]["release_pairs"])] if "release_pairs" in config["layout"] else []
        self.chokepoints = [int(x) for x in st["chokepoints"]] if "chokepoints" in st else []
        
        self.sinks_pi = np.array([float(x) for x in st["sinks"]["pi"]], dtype=float)

    def act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        
        c = np.array(obs["graph_now.c"], dtype=float)
        tariff = np.array(obs["graph_now.tariff"], dtype=float)
        war_risk = np.array(obs["graph_now.war_risk"], dtype=int)
        
        stock = np.array(obs["stock.qty"], dtype=float)
        backlog = np.array(obs["backlog.qty"], dtype=float)
        demand_forecast = np.array(obs["demand_forecast.qty"], dtype=float)
        queue_lots = np.array(obs["queue_lots.qty"], dtype=float) if "queue_lots.qty" in obs else None
        
        mask = np.array(obs["action_mask"], dtype=float)
        override_mask = np.array(obs["override_mask"], dtype=float) if "override_mask" in obs else np.ones(self.n_over)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Slot capacity
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None:
                c_slot = cap[self.slot_edge[i]]
            else:
                c_slot = min(cap[e] for e in self.lane_edges[lane]) if self.lane_edges[lane] else 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Compute cost + risk per slot
        slot_cost = np.zeros(n)
        slot_risk = np.zeros(n)
        for i in range(n):
            edge_idx = self.slot_edge[i]
            k = self.slot_k[i]
            cost_val = c[edge_idx] + tariff[edge_idx, k]
            slot_cost[i] = max(cost_val, 0.0)
            slot_risk[i] = 0.0
        
        # Compute demand pressure
        demand_pressure = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]]
            k = self.slot_k[i]
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                press = backlog[d_idx] + sum(demand_forecast[d_idx, h] for h in range(min(3, demand_forecast.shape[1])))
                demand_pressure[i] = press
        
        # Group by source
        groups = {}
        for i in range(n):
            key = (int(self.edge_tail[self.slot_edge[i]]), int(self.slot_k[i]))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        # Allocate flows
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            idxs_sorted = sorted(idxs, key=lambda i: (-demand_pressure[i], slot_cost[i]))
            
            remaining = avail
            for i in idxs_sorted:
                slot_request = min(slot_cap[i], remaining)
                flows[i] = slot_request
                remaining -= slot_request
        
        flows = flows * mask
        
        # Tanker release strategy
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        # Simple strategy: hold tanker cargo at high-risk chokepoints
        for j in range(self.n_rel):
            if j < len(self.override_chokepoint):
                chp = self.override_chokepoint[j]
                chp_idx = self.chokepoints.index(chp) if chp in self.chokepoints else -1
                if chp_idx >= 0 and war_risk[chp_idx] > 0:
                    release_mode[j] = 2  # hold
                else:
                    release_mode[j] = 0  # default release
        
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}