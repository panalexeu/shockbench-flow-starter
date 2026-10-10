# -999
# error: 832 of 832 weeks crashed, first: episode 0, week 0: TypeError: string indices must be integers, not 'str'
import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Extract layouts and static info
        self.stock_slots = config['layout']['stock_slots']
        self.action_slots = config['static']['action_slots']
        self.demands = config['layout']['demands']
        self.supply_slots = config['layout']['supply_slots']
        self.override_slots = config['static']['override_slots']
        self.release_pairs = config['layout']['release_pairs']
        self.chokepoints = config['layout']['chokepoints']
        
        # Build maps for lookups
        self.slot_to_idx = {}
        for i, slot in enumerate(self.action_slots):
            key = (slot['edge'], slot['k'], slot.get('lane'))
            self.slot_to_idx[key] = i
        
        self.stock_to_idx = {}
        for i, (node, k) in enumerate(self.stock_slots):
            self.stock_to_idx[(node, k)] = i
        
        self.demand_to_idx = {}
        for i, (node, k) in enumerate(self.demands):
            self.demand_to_idx[(node, k)] = i
        
        # Random seed
        np.random.seed(config['policy_seed'])
    
    def act(self, observation):
        flows = np.zeros(len(self.action_slots))
        
        # Get current week
        week = int(observation['week'][0])
        
        # Extract relevant observations
        stock_qty = observation['stock.qty']
        stock_observed = observation['stock.qty.observed']
        backlog = observation['backlog.qty']
        demand_forecast = observation['demand_forecast.qty']
        graph_prohibited = observation['graph_now.prohibited']
        graph_capacity = observation['graph_now.u']
        graph_open = observation['graph_now.open']
        action_mask = observation['action_mask']
        slot_mask = observation['slot_mask']
        
        # Simple strategy: try to fulfill next week's demand from available stock
        # Look at demand forecast for week t (column 0 of demand_forecast)
        demand_this_week = demand_forecast[:, 0] if demand_forecast.size > 0 else np.zeros(len(self.demands))
        
        # For each sink (demand location), calculate what we need to send
        for demand_idx, demand_val in enumerate(demand_this_week):
            if demand_val <= 0 or demand_val != demand_val:  # Skip invalid
                continue
            
            sink_node, sink_k = self.demands[demand_idx]
            
            # Find stock at this node for this commodity
            stock_idx = self.stock_to_idx.get((sink_node, sink_k))
            if stock_idx is not None and stock_observed[stock_idx] > 0:
                # Already have stock at sink, don't need to dispatch
                continue
            
            # Look for supply routes that can deliver this commodity
            demand_needed = max(0, demand_val - backlog[demand_idx])
            if demand_needed <= 0.01:
                continue
            
            # Find action slots that could deliver to this sink
            for slot_idx, slot in enumerate(self.action_slots):
                if slot['k'] != sink_k:
                    continue
                if slot_mask[slot_idx] > 0:  # Prohibited
                    continue
                if action_mask[slot_idx] == 0:  # Masked out
                    continue
                
                edge_head = self.config['static']['edges']['head'][slot['edge']]
                if edge_head == sink_node:
                    # This slot delivers to our sink
                    # Check available stock
                    edge_tail = self.config['static']['edges']['tail'][slot['edge']]
                    available_stock_idx = self.stock_to_idx.get((edge_tail, sink_k))
                    
                    if available_stock_idx is not None and stock_observed[available_stock_idx] > 0:
                        available = stock_qty[available_stock_idx]
                        cap = graph_capacity[slot['edge']]
                        
                        # Check if edge is open
                        edge_mode = self.config['static']['edges']['mode'][slot['edge']]
                        for choke_idx, choke_node in enumerate(self.chokepoints):
                            if edge_head == choke_node or edge_tail == choke_node:
                                if graph_open[choke_idx] < 1.0:
                                    cap *= graph_open[choke_idx]
                        
                        flow = min(available, cap, demand_needed)
                        if flow > 0.01:
                            flows[slot_idx] = flow
                            demand_needed -= flow
        
        # For tanker cargo at chokepoints: use default release (mode 0)
        override_qty = np.zeros(len(self.override_slots))
        release_mode = np.zeros(len(self.release_pairs), dtype=int)
        
        action = {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }
        
        return action