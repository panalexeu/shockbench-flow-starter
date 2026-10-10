# -999
# error: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
import numpy as np

class Agent:
    """Disruption-responsive: react to warnings and chokepoint closures."""
    
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
        self.layout = layout
        self.chokepoints = layout["chokepoints"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = np.zeros_like(self.capacity)
        action_mask = observation["action_mask"]
        
        # Get disruption signals
        warning_score = observation["warning.score"]  # (num_warning_units,)
        closure_end = observation["closure_end.chokepoint"]
        closure_observed = observation["closure_end.chokepoint"].any()  # simplified
        graph_open = observation["graph_now.open"]  # (num_chokepoints,)
        
        # Compute max warning intensity
        max_warning = np.max(warning_score) if warning_score.size > 0 else 0.0
        
        # Identify closed or at-risk chokepoints
        closed_chokepoints = set()
        if closure_end.size > 0:
            for cp_idx in closure_end:
                if cp_idx >= 0:
                    closed_chokepoints.add(int(cp_idx))
        
        at_risk_chokepoints = set()
        for cp_idx, open_frac in enumerate(graph_open):
            if open_frac < 0.5:  # Less than 50% open
                at_risk_chokepoints.add(cp_idx)
        
        # Build high-urgency flows
        demand_forecast = observation["demand_forecast.qty"]
        demand_observed = observation["demand_forecast.qty.observed"]
        backlog = observation["backlog.qty"]
        stock = observation["stock.qty"]
        
        # Compute demand urgency
        num_commodities = len(self.commodities["id"])
        urgency = np.zeros(num_commodities)
        
        if demand_observed.sum() > 0:
            for sink_idx, d_row in enumerate(demand_forecast):
                if demand_observed[sink_idx]:
                    k_idx = self.sinks["k"][sink_idx]
                    urgency[k_idx] += d_row[0]
        
        # Add backlog penalty
        for sink_idx in range(len(self.sinks["k"])):
            if sink_idx < len(backlog):
                k_idx = self.sinks["k"][sink_idx]
                urgency[k_idx] += backlog[sink_idx]
        
        # Allocate flows with disruption awareness
        total_urgency = np.sum(urgency) + 1e-6
        
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            # Base allocation from urgency
            if total_urgency > 0:
                base_factor = 0.7 + 0.3 * (urgency[k_idx] / total_urgency)
            else:
                base_factor = 0.7
            
            # Boost for high warnings: send preemptively through at-risk routes
            if max_warning > 0.5:
                base_factor = min(base_factor + 0.2, 1.0)
            
            flows[slot_idx] = self.capacity[slot_idx] * base_factor
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }
