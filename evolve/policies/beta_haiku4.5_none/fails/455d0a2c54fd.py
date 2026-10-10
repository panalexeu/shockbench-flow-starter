# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Threat-aware + chokepoint override strategy."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        layout = config["layout"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.commodities = static["commodities"]
        self.sinks = static["sinks"]
        self.override_slots = static["override_slots"]
        self.chokepoints_layout = layout["chokepoints"]
        self.release_pairs_layout = layout["release_pairs"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        week = int(observation["week"][0])
        
        # === DETECT THREATS ===
        pending_edge = observation["pending_prohibitions.edge"]
        pending_k = observation["pending_prohibitions.k"]
        pending_week = observation["pending_prohibitions.effective_week"]
        pending_obs = observation["pending_prohibitions.edge.observed"]
        
        threat_boost = np.ones(len(flows))
        threatened_edges = {}
        
        if pending_obs.any():
            for i in np.where(pending_obs)[0]:
                edge_idx = int(pending_edge[i])
                k_idx = int(pending_k[i])
                eff_week = int(pending_week[i])
                weeks_until = max(0, eff_week - week)
                
                # Boost based on imminence
                if 0 < weeks_until <= 1:
                    boost = 1.5
                elif 1 < weeks_until <= 2:
                    boost = 1.35
                elif 2 < weeks_until <= 4:
                    boost = 1.15
                else:
                    boost = 1.0
                
                if boost > 1.0:
                    threatened_edges[(edge_idx, k_idx)] = boost
                    # Apply to all slots using this edge and commodity
                    for slot_idx in range(len(flows)):
                        if (self.action_slots["edge"][slot_idx] == edge_idx and
                            self.action_slots["k"][slot_idx] == k_idx):
                            threat_boost[slot_idx] = max(threat_boost[slot_idx], boost)
        
        flows = flows * threat_boost * action_mask
        
        # === COMPUTE URGENCY FOR OVERRIDES ===
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        demand_forecast = observation["demand_forecast.qty"]
        demand_forecast_obs = observation["demand_forecast.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                urgency[k_idx] += backlog[sink_idx]
            if sink_idx < len(demand_forecast_obs) and demand_forecast_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                urgency[k_idx] += demand_forecast[sink_idx, 0]
        
        max_urg = np.max(urgency) if np.max(urgency) > 0 else 1.0
        urgency_norm = urgency / max_urg
        
        # === CHOKEPOINT OVERRIDES ===
        graph_open = observation["graph_now.open"]
        override_mask = observation["override_mask"]
        
        # Map chokepoint node to index in graph_now.open
        cp_node_to_idx = {}
        for cp_idx, cp_node in enumerate(self.chokepoints_layout):
            cp_node_to_idx[cp_node] = cp_idx
        
        for override_idx in range(len(self.override_slots["chokepoint"])):
            if not override_mask[override_idx]:
                continue
            
            cp_node = self.override_slots["chokepoint"][override_idx]
            k_idx = self.override_slots["k"][override_idx]
            
            # Check if chokepoint is constrained and commodity is urgent
            if cp_node in cp_node_to_idx:
                cp_idx = cp_node_to_idx[cp_node]
                if graph_open[cp_idx] < 0.7 and urgency_norm[k_idx] > 0.6:
                    # Find release_mode index for this (chokepoint, k) pair
                    for rp_idx, rp in enumerate(self.release_pairs_layout):
                        if rp[0] == cp_node and rp[1] == k_idx:
                            self.release_mode[rp_idx] = 1
                            # Set override quantity moderately
                            self.override_qty[override_idx] = 0.6 * self.capacity[min(override_idx, len(self.capacity) - 1)]
                            break
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }