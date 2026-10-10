# 0.28909448691824724
import numpy as np

class Agent:
    """Inventory-balanced demand response: send to where stock is low, reduce where high."""
    
    def __init__(self, config=None):
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
        self.layout = layout
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        
        # Stock levels: estimate per destination node
        stock_qty = observation["stock.qty"]
        stock_obs = observation["stock.qty.observed"]
        
        stock_by_node_k = {}
        if stock_obs.sum() > 0:
            for stock_slot_idx in np.where(stock_obs)[0]:
                if stock_slot_idx < len(self.layout["stock_slots"]):
                    node_idx, k_idx = self.layout["stock_slots"][stock_slot_idx]
                    stock_by_node_k[(node_idx, k_idx)] = float(stock_qty[stock_slot_idx])
        
        # Demand signals
        demand_forecast = observation["demand_forecast.qty"]
        backlog = observation["backlog.qty"]
        backlog_obs = observation["backlog.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        commodity_urgency = np.zeros(num_commodities)
        
        for sink_idx in range(len(self.sinks["k"])):
            k_idx = self.sinks["k"][sink_idx]
            if sink_idx < len(backlog_obs) and backlog_obs[sink_idx]:
                commodity_urgency[k_idx] += float(backlog[sink_idx]) * 2.5
            if sink_idx < len(demand_forecast):
                commodity_urgency[k_idx] += float(demand_forecast[sink_idx, 0])
        
        # Compute average stock per commodity for reference
        avg_stock_by_k = {}
        for k_idx in range(num_commodities):
            stocks = [s for (n, k), s in stock_by_node_k.items() if k == k_idx]
            avg_stock_by_k[k_idx] = np.mean(stocks) if stocks else 1.0
        
        # Apply allocation
        max_urgency = np.max(commodity_urgency) if np.max(commodity_urgency) > 0 else 1.0
        
        for slot_idx in range(len(flows)):
            if not action_mask[slot_idx]:
                flows[slot_idx] = 0
                continue
            
            k_idx = self.action_slots["k"][slot_idx]
            edge_idx = self.action_slots["edge"][slot_idx]
            
            # Urgency factor: 0.8 to 1.15
            urgency_norm = commodity_urgency[k_idx] / (max_urgency + 1e-9)
            urgency_factor = 0.8 + 0.35 * min(urgency_norm, 1.0)
            
            # Destination stock factor: if destination has high stock, reduce send
            dest_node = int(self.edges["head"][edge_idx]) if edge_idx < len(self.edges["head"]) else -1
            stock_factor = 1.0
            
            if (dest_node, k_idx) in stock_by_node_k:
                dest_stock = stock_by_node_k[(dest_node, k_idx)]
                avg_stock = avg_stock_by_k.get(k_idx, 1.0)
                # If stock > 1.3x average, reduce to 0.85; if < 0.7x average, boost to 1.05
                if dest_stock > avg_stock * 1.3:
                    stock_factor = 0.85
                elif dest_stock < avg_stock * 0.7:
                    stock_factor = 1.05
            
            flows[slot_idx] *= (urgency_factor * stock_factor)
        
        return {
            "flows": flows,
            "override_qty": self.override_qty,
            "release_mode": self.release_mode
        }