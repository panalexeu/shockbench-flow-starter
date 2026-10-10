# 0.5328712851687165
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
        
        self.stock_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["stock_slots"]):
            self.stock_idx[(int(nd), int(k))] = r
        
        self.demand_idx = {}
        for r, (nd, k) in enumerate(config["layout"]["demands"]):
            self.demand_idx[(int(nd), int(k))] = r
        
        # Lot key tracking
        self.lot_keys = config["layout"]["lot_keys"] if "lot_keys" in config["layout"] else []
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
        
        # Queue lots to detect chokepoint congestion
        queue_congestion = {}
        if "queue_lots.qty" in obs:
            queue_lots = np.array(obs["queue_lots.qty"], dtype=float)
            queue_observed = np.array(obs["queue_lots.qty.observed"]) if "queue_lots.qty.observed" in obs else np.ones_like(queue_lots)
            # Sum congestion per lot_key (chokepoint + commodity + lane + next_edge)
            for lot_idx in range(min(len(queue_lots), len(self.lot_keys))):
                if queue_observed[lot_idx].sum() > 0:  # Has waiting lots
                    lot_key = self.lot_keys[lot_idx] if lot_idx < len(self.lot_keys) else None
                    if lot_key:
                        chp = lot_key[0]  # chokepoint is first element
                        queue_congestion[chp] = queue_congestion.get(chp, 0.0) + queue_lots[lot_idx].sum()
        
        mask = np.array(obs["action_mask"], dtype=float)
        
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
                for h in range(min(3, demand_forecast.shape[1])):
                    press += demand_forecast[d_idx, h] * (1.0 - 0.2 * h)
                demand_pressure[i] = press
        
        # Penalize routes passing through congested chokepoints
        for i in range(n):
            lane = self.slot_lane[i]
            if lane is not None and lane < len(self.lane_edges):
                # Check if any chokepoint on this lane is congested
                for chp in self.chokepoints:
                    if chp in queue_congestion and queue_congestion[chp] > 10.0:
                        demand_pressure[i] *= 0.7  # Reduce incentive to use congested routes
        
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
        
        return {"flows": flows, "override_qty": np.zeros(self.n_over), "release_mode": np.zeros(self.n_rel, dtype=np.int64)}