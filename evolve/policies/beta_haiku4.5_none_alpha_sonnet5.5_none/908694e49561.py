# 0.5328759942236484
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
        self.slot_edge = list(st["action_slots"]["edge"])
        self.slot_k = list(st["action_slots"]["k"])
        self.slot_lane = list(st["action_slots"]["lane"])
        self.lane_edges = st["lanes"]["edges"]
        self.edge_head = list(edges["head"])
        self.edge_tail = list(edges["tail"])
        
        # Override and release pair indexing
        self.override_chokepoint = [int(x) for x in st["override_slots"]["chokepoint"]]
        self.override_k = [int(x) for x in st["override_slots"]["k"]]
        self.override_out_edge = [int(x) for x in st["override_slots"]["out_edge"]]
        self.release_pair_chp = [int(config["layout"]["release_pairs"][j][0]) for j in range(self.n_rel)]
        self.release_pair_k = [int(config["layout"]["release_pairs"][j][1]) for j in range(self.n_rel)]
        
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        
        self.chokepoints = [int(x) for x in st["chokepoints"]] if "chokepoints" in st else []

    def act(self, obs):
        u = np.array(obs["graph_now.u"], dtype=float)
        uo = np.array(obs["graph_now.u.observed"]) if "graph_now.u.observed" in obs else np.ones_like(u)
        cap = np.where(uo > 0, u, self.u0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        cap = np.clip(cap, 0, None)
        
        stock = np.array(obs["stock.qty"], dtype=float)
        stock = np.clip(stock, 0, None)
        
        backlog = np.array(obs["backlog.qty"], dtype=float)
        backlog = np.clip(backlog, 0, None)
        
        demand_forecast = np.array(obs["demand_forecast.qty"], dtype=float)
        demand_forecast = np.clip(demand_forecast, 0, None)
        
        war_risk = np.array(obs["graph_now.war_risk"], dtype=int) if "graph_now.war_risk" in obs else np.zeros(len(self.chokepoints), dtype=int)
        open_frac = np.array(obs["graph_now.open"], dtype=float) if "graph_now.open" in obs else np.ones(len(self.chokepoints), dtype=float)
        
        # Queue lots for chokepoint status
        queue_at_chp = np.zeros(len(self.chokepoints))
        if "queue_lots.qty" in obs:
            queue_lots = np.array(obs["queue_lots.qty"], dtype=float)
            queue_observed = np.array(obs["queue_lots.qty.observed"]) if "queue_lots.qty.observed" in obs else np.ones_like(queue_lots)
            # Sum per lot_key row (first element is chokepoint)
            for lot_idx in range(len(queue_lots)):
                if queue_observed[lot_idx].any():
                    total_at_lot = queue_lots[lot_idx].sum()
                    if total_at_lot > 0:
                        # Find chokepoint index for this lot; for now assume simple indexing
                        if lot_idx < len(self.chokepoints):
                            queue_at_chp[lot_idx] += total_at_lot
        
        mask = np.array(obs["action_mask"], dtype=float)
        override_mask = np.array(obs["override_mask"], dtype=int) if "override_mask" in obs else np.ones(self.n_over, dtype=int)
        
        n = len(self.slot_edge)
        flows = np.zeros(n)
        
        # Slot capacity
        slot_cap = np.zeros(n)
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is None or lane >= len(self.lane_edges):
                c_slot = cap[self.slot_edge[i]] if self.slot_edge[i] < len(cap) else 0.0
            else:
                lane_edges = self.lane_edges[lane]
                c_slot = min((cap[e] for e in lane_edges if e < len(cap)), default=0.0) if lane_edges else 0.0
            slot_cap[i] = max(c_slot, 0.0)
        
        # Demand pressure
        demand_pressure = np.zeros(n)
        for i in range(n):
            edge_head = self.edge_head[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_head) else -1
            k = self.slot_k[i]
            key = (int(edge_head), int(k))
            if key in self.demand_idx:
                d_idx = self.demand_idx[key]
                press = backlog[d_idx] * 2.0
                for h in range(min(4, demand_forecast.shape[1])):
                    press += demand_forecast[d_idx, h] * (1.0 - 0.15 * h)
                demand_pressure[i] = press
        
        # Group and allocate
        groups = {}
        for i in range(n):
            tail_node = self.edge_tail[self.slot_edge[i]] if self.slot_edge[i] < len(self.edge_tail) else -1
            k = self.slot_k[i]
            key = (int(tail_node), int(k))
            if key in self.stock_idx:
                groups.setdefault(key, []).append(i)
        
        for key, idxs in groups.items():
            avail = max(stock[self.stock_idx[key]], 0.0)
            idxs_sorted = sorted(idxs, key=lambda i: -demand_pressure[i])
            
            remaining = avail
            for i in idxs_sorted:
                slot_request = min(slot_cap[i], remaining)
                flows[i] = slot_request
                remaining -= slot_request
        
        flows = flows * mask
        
        # Strategic tanker release: release from low-risk/uncongested chokepoints
        override_qty = np.zeros(self.n_over)
        release_mode = np.zeros(self.n_rel, dtype=np.int64)
        
        for j in range(self.n_rel):
            chp_idx = -1
            for c_idx, chp in enumerate(self.chokepoints):
                if chp == self.release_pair_chp[j]:
                    chp_idx = c_idx
                    break
            
            if chp_idx >= 0:
                # High war risk or very congested: hold
                if war_risk[chp_idx] > 1 or queue_at_chp[chp_idx] > 20.0:
                    release_mode[j] = 2
                # Moderate risk or some congestion: default release
                elif war_risk[chp_idx] == 1 or queue_at_chp[chp_idx] > 5.0:
                    release_mode[j] = 0
                # Low risk and uncongested: default release
                else:
                    release_mode[j] = 0
            else:
                release_mode[j] = 0
        
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}