# 0.4699062375097181
import numpy as np

class Agent:
    """Refined full-capacity baseline with smart chokepoint overrides."""
    
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
        self.override_slots = static["override_slots"]
        self.layout_chokepoints = layout["chokepoints"]
        self.layout_release_pairs = layout["release_pairs"]
        self.layout_demands = layout["demands"]
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        # Core strategy: send full capacity on allowed slots
        flows = self.capacity * observation["action_mask"]
        
        # Chokepoint override logic: release tanker cargo when throughput is constrained
        graph_open = observation["graph_now.open"]
        graph_kappa_tb = observation["graph_now.kappa.tb"]
        backlog = observation["backlog.qty"]
        demand_forecast = observation["demand_forecast.qty"]
        
        # Assess urgency per commodity
        num_commodities = len(self.commodities["id"])
        commodity_urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.layout_demands)):
            if sink_idx < len(backlog):
                k = self.layout_demands[sink_idx][1]
                commodity_urgency[k] += backlog[sink_idx]
            if sink_idx < len(demand_forecast):
                k = self.layout_demands[sink_idx][1]
                commodity_urgency[k] += np.sum(demand_forecast[sink_idx, :2])
        
        # Normalize urgency
        if np.max(commodity_urgency) > 0:
            commodity_urgency = commodity_urgency / np.max(commodity_urgency)
        
        # Identify constrained chokepoints
        constrained_chokepoints = graph_open < 0.7
        
        # Deploy overrides on constrained chokepoints with high urgency
        for override_idx in range(len(self.override_slots["chokepoint"])):
            if observation["override_mask"][override_idx] == 0:
                continue
            
            cp_idx = self.override_slots["chokepoint"][override_idx]
            k_idx = self.override_slots["k"][override_idx]
            
            # Use override if:
            # 1. Commodity has high urgency
            # 2. Chokepoint is constrained
            # 3. Throughput is low
            if (commodity_urgency[k_idx] > 0.6 and 
                cp_idx < len(constrained_chokepoints) and 
                constrained_chokepoints[cp_idx] and
                cp_idx < len(graph_kappa_tb) and
                graph_kappa_tb[cp_idx] < 0.8):
                
                # Release override cargo
                self.release_mode[override_idx] = 1
                # Send proportional to urgency and available capacity
                self.override_qty[override_idx] = 50.0 * commodity_urgency[k_idx]
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }