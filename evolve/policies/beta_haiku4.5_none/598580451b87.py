# -0.6471418277556716
import numpy as np

class Agent:
    """Smooth cost-aware capacity allocation with backlog signal."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.action_slots = action_slots
        self.sinks = static["sinks"]
        self.commodities = static["commodities"]
        self.edges = static["edges"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        flows = flows * action_mask
        
        # Compute backlog pressure per commodity
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(min(len(backlog), len(self.sinks["k"]))):
            k_idx = self.sinks["k"][sink_idx]
            backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Normalize backlog pressure
        max_backlog = backlog_per_k.max()
        if max_backlog > 1e-6:
            backlog_pressure = backlog_per_k / max_backlog
        else:
            backlog_pressure = np.ones(num_commodities) / num_commodities
        
        # Get freight costs
        graph_c = observation["graph_now.c"]
        graph_c_obs = observation["graph_now.c.observed"]
        
        # Get tariffs
        graph_tariff = observation["graph_now.tariff"]
        tariff_obs = observation["graph_now.tariff.observed"]
        
        # Compute cost metrics for smooth penalty
        valid_costs = graph_c[graph_c_obs.astype(bool)]
        if len(valid_costs) > 0:
            mean_cost = np.mean(valid_costs)
            std_cost = np.std(valid_costs)
            cost_p75 = np.percentile(valid_costs, 75)
        else:
            mean_cost = 1.0
            std_cost = 0.1
            cost_p75 = 1.5
        
        # Apply smooth allocation: backlog boost + cost penalty
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx] <= 0:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Base allocation: scaled by commodity backlog pressure
            allocation = 0.6 + 0.4 * backlog_pressure[k_idx]
            
            # Smooth cost penalty: sigmoid-like curve
            if graph_c_obs[edge_idx] > 0:
                edge_cost = float(graph_c[edge_idx])
                cost_ratio = (edge_cost - mean_cost) / (std_cost + 1e-6)
                cost_penalty = 1.0 / (1.0 + np.exp(cost_ratio * 0.5))  # Sigmoid: smooth decay
                allocation *= cost_penalty
            
            # Smooth tariff penalty
            if tariff_obs[edge_idx, k_idx] > 0:
                tariff_rate = float(graph_tariff[edge_idx, k_idx])
                tariff_penalty = np.exp(-tariff_rate * 2.0)  # Exponential decay with tariff
                allocation *= tariff_penalty
            
            flows[slot_idx] = self.capacity[slot_idx] * allocation
        
        # Normalize to prevent excessive over-allocation
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.15:
            flows = flows * (total_allowed * 1.1 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
