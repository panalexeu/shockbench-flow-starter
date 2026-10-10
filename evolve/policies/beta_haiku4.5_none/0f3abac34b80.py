# 0.4699062375097186
import numpy as np

class Agent:
    """Enhanced baseline with modest backlog boost and disruption awareness."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.edges = static["edges"]
        self.override_slots = static.get("override_slots", {})
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        # Compute backlog per commodity (main urgency signal)
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(min(len(backlog), len(self.sinks["k"]))):
            k_idx = self.sinks["k"][sink_idx]
            backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Normalize urgency
        max_backlog = backlog_per_k.max()
        if max_backlog > 1e-6:
            urgency = backlog_per_k / max_backlog
        else:
            urgency = np.ones(num_commodities) / num_commodities
        
        # Check for pending prohibitions (routes under threat)
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threatened_routes = set()
        threat_deadline = {}
        obs_indices = np.where(pending_obs)[0]
        for i in obs_indices:
            edge_idx = int(pending_edge[i])
            k_idx = int(pending_k[i])
            eff_week = int(pending_week[i])
            key = (edge_idx, k_idx)
            threatened_routes.add(key)
            threat_deadline[key] = eff_week
        
        # Apply flows with backlog boost and threat avoidance
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx] <= 0:
                flows[slot_idx] = 0
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Base: full capacity
            allocation = 1.0
            
            # Boost on high-urgency commodities
            allocation += 0.25 * urgency[k_idx]
            
            # Threatened routes: accelerate shipment (1.15x boost if threat within 3 weeks)
            route_key = (edge_idx, k_idx)
            if route_key in threatened_routes:
                weeks_until = threat_deadline[route_key] - week
                if 0 < weeks_until <= 3:
                    allocation *= 1.15  # Push more before closure
            
            flows[slot_idx] = self.capacity[slot_idx] * allocation
        
        flows = flows * action_mask
        
        # Normalize if over-allocated
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.25:
            flows = flows * (total_allowed * 1.2 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
