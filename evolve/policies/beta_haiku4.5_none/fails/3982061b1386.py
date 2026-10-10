# -999
# error: 832 of 832 weeks crashed, first: episode 0, week 0: KeyError: 'stock_slots'
import numpy as np

class Agent:
    """Baseline+: full capacity, reduced when destination stock is high."""
    
    def __init__(self, config=None):
        self.config = config
        static = config["static"]
        action_space = config["spaces"]["action"]
        
        u0 = static["edges"]["u0"]
        action_slots = static["action_slots"]
        self.capacity = np.array([u0[e] if u0[e] is not None else 1e9 for e in action_slots["edge"]], dtype=float)
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
        
        self.action_slots = action_slots
        self.edges = static["edges"]
        self.stock_slots = static["stock_slots"]
    
    def act(self, observation):
        flows = self.capacity * observation["action_mask"]
        
        # Check stock levels
        stock = observation["stock.qty"]  # (num_stock_slots,)
        stock_observed = observation["stock.qty.observed"]
        
        # Compute average stock level
        if stock_observed.sum() > 0:
            avg_stock = np.sum(stock * stock_observed) / (stock_observed.sum() + 1e-6)
            high_stock_threshold = avg_stock * 2.0
        else:
            high_stock_threshold = 1e6
        
        # Get routing information from edges
        edge_heads = self.edges["head"]  # destination node of each edge
        
        # For each action slot, check if destination has high stock
        for slot_idx in range(len(flows)):
            if flows[slot_idx] == 0:
                continue
            
            edge_idx = self.action_slots["edge"][slot_idx]
            k_idx = self.action_slots["k"][slot_idx]
            
            if edge_idx >= len(edge_heads):
                continue
            
            dest_node = int(edge_heads[edge_idx])
            
            # Find stock slot for (dest_node, k_idx)
            high_stock_found = False
            for stock_slot_idx in range(len(self.stock_slots["node"])):
                if (int(self.stock_slots["node"][stock_slot_idx]) == dest_node and 
                    int(self.stock_slots["commodity"][stock_slot_idx]) == k_idx):
                    if stock_observed[stock_slot_idx]:
                        if float(stock[stock_slot_idx]) > high_stock_threshold:
                            high_stock_found = True
                    break
            
            # Reduce flow if destination has excess stock
            if high_stock_found:
                flows[slot_idx] *= 0.7
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }