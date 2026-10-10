# 0.4699062375097181
import numpy as np

class Agent:
    """Stock-aware aggressive backlog with pipeline timing: reduce urgency if stock+pipeline sufficient."""
    
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
        
        # Compute backlog per commodity
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Compute stock + near-term pipeline per commodity (arrivals within 2 weeks)
        stock = observation["stock.qty"]
        stock_obs = observation["stock.qty.observed"]
        stock_per_k = np.zeros(num_commodities)
        
        # Aggregate stock by commodity (map stock_slots to commodities)
        for stock_idx in range(len(stock_obs)):
            if stock_obs[stock_idx]:
                # Safely extract commodity if layout provides it
                if hasattr(self.config.get("layout", {}), "get") and "stock_slots" in self.config.get("layout", {}):
                    k_idx = self.config["layout"]["stock_slots"][stock_idx][1]
                else:
                    k_idx = stock_idx % num_commodities
                if k_idx < num_commodities:
                    stock_per_k[k_idx] += stock[stock_idx]
        
        pipeline_qty = observation["pipeline.qty"]
        pipeline_k = observation["pipeline.k"]
        pipeline_arrival = observation["pipeline.arrival_week"]
        pipeline_obs = observation["pipeline.qty.observed"]
        current_week = int(observation["week"][0])
        
        pipeline_per_k = np.zeros(num_commodities)
        if pipeline_obs.any():
            for i in np.where(pipeline_obs)[0]:
                k_idx = int(pipeline_k[i])
                arrival = int(pipeline_arrival[i])
                # Include shipments arriving in next 2 weeks
                if arrival <= current_week + 2 and k_idx < num_commodities:
                    pipeline_per_k[k_idx] += pipeline_qty[i]
        
        # Net urgency: backlog minus available inventory
        available_inventory = stock_per_k + pipeline_per_k
        net_urgency = np.maximum(0, backlog_per_k - 0.7 * available_inventory)
        
        max_urgency = np.max(net_urgency) if np.max(net_urgency) > 0 else 1.0
        urgency_norm = net_urgency / (max_urgency + 1e-9)
        
        # Apply aggressive boost: 1.0 + up to 0.40x on urgency
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Backlog boost
            backlog_boost = 1.0 + 0.40 * urgency_norm[k_idx]
            
            # Lead-time boost (only when both backlog and urgency are high)
            lead_time_boost = 1.0
            if urgency_norm[k_idx] > 0.2 and edge_idx < len(self.edges["tau0"]):
                tau = self.edges["tau0"][edge_idx]
                if tau > 2:
                    tau_factor = min((tau - 1.0) / 5.0, 1.0)
                    lead_time_boost = 1.0 + 0.12 * tau_factor
            
            flows[slot_idx] *= (backlog_boost * lead_time_boost)
        
        # Hard cap to stay within reasonable bounds
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.2:
            flows = flows * (total_allowed * 1.2 / (total_flows + 1e-9))
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }