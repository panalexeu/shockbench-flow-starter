# 0.16744218658598117
import numpy as np

class Agent:
    """Balanced backlog + pipeline awareness: avoid over-committing when pipeline is full."""
    
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
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Backlog per commodity
        backlog = observation["backlog.qty"]
        num_commodities = len(self.commodities["id"])
        backlog_per_commodity = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                backlog_per_commodity[k_idx] += backlog[sink_idx]
        
        # Pipeline inventory per commodity (nearby arrivals: current + next 2 weeks)
        pipeline_qty = observation["pipeline.qty"]
        pipeline_k = observation["pipeline.k"]
        pipeline_arrival = observation["pipeline.arrival_week"]
        pipeline_obs = observation["pipeline.qty.observed"]
        current_week = int(observation["week"][0])
        
        pipeline_per_commodity = np.zeros(num_commodities)
        if pipeline_obs.any():
            for i in np.where(pipeline_obs)[0]:
                k_idx = int(pipeline_k[i])
                arrival = int(pipeline_arrival[i])
                if arrival <= current_week + 2:
                    pipeline_per_commodity[k_idx] += pipeline_qty[i]
        
        # Compute effective need: backlog + upcoming demand minus pipeline
        stock = observation["stock.qty"]
        stock_obs = observation["stock.qty.observed"]
        stock_per_commodity = np.zeros(num_commodities)
        
        # Aggregate stock by commodity (approx: use first few stocks)
        stock_idx = 0
        for k_idx in range(min(num_commodities, len(self.sinks["k"]))):
            if stock_idx < len(stock_obs) and stock_obs[stock_idx]:
                stock_per_commodity[k_idx] += stock[stock_idx]
            stock_idx += 1
        
        # Need: backlog dominates, reduced by available stock + pipeline
        total_available = stock_per_commodity + pipeline_per_commodity
        urgency = np.maximum(0, backlog_per_commodity - 0.5 * total_available)
        max_urgency = np.max(urgency)
        
        if max_urgency > 1e-6:
            urgency_normalized = urgency / (max_urgency + 1e-9)
        else:
            urgency_normalized = np.ones(num_commodities) / num_commodities
        
        # Conservative ramp: 0.7x to 1.1x based on urgency
        flows = flows * action_mask
        for slot_idx in range(len(flows)):
            k_idx = self.action_slots["k"][slot_idx]
            allocation = 0.7 + 0.4 * urgency_normalized[k_idx]
            flows[slot_idx] *= allocation
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }