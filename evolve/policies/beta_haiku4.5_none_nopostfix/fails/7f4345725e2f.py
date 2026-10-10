# -999
# error: 832 of 832 weeks crashed, first: episode 0, week 0: ValueError: Seed must be between 0 and 2**32 - 1
import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        np.random.seed(config['policy_seed'])
        
        # Extract static structures (all are dicts of parallel lists)
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.nodes = config['static']['nodes']
        self.commodities = config['static']['commodities']
        
        # Extract layouts
        self.stock_slots = config['layout']['stock_slots']
        self.supply_slots = config['layout']['supply_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        
        # Build reverse maps
        self.stock_slot_map = {tuple(s): i for i, s in enumerate(self.stock_slots)}
        self.demand_map = {tuple(d): i for i, d in enumerate(self.demands)}
        
    def act(self, observation):
        week = int(observation['week'][0])
        
        # Initialize action
        num_action_slots = len(self.action_slots['edge'])
        num_override_slots = len(self.override_slots['chokepoint'])
        num_release_pairs = len(self.release_pairs)
        
        flows = np.zeros(num_action_slots, dtype=np.float64)
        override_qty = np.zeros(num_override_slots, dtype=np.float64)
        release_mode = np.zeros(num_release_pairs, dtype=np.int64)
        
        try:
            # Extract observations
            stock_qty = observation['stock.qty']
            stock_observed = observation['stock.qty.observed']
            demand_forecast = observation['demand_forecast.qty']
            graph_capacity = observation['graph_now.u']
            graph_open = observation['graph_now.open']
            graph_prohibited = observation['graph_now.prohibited']
            action_mask = observation['action_mask']
            
            # Get demand for current week (column 0 of forecast)
            demand_now = demand_forecast[:, 0] if demand_forecast.ndim > 1 else demand_forecast
            
            # Simple greedy strategy: for each demand, try to route available supply
            for demand_idx in range(len(self.demands)):
                demand_qty = demand_now[demand_idx]
                if demand_qty <= 0 or not np.isfinite(demand_qty):
                    continue
                    
                sink_node, sink_k = self.demands[demand_idx]
                demand_needed = demand_qty
                
                # Try to fulfill from available stocks via action slots
                for slot_idx in range(num_action_slots):
                    if demand_needed <= 0.01:
                        break
                    
                    # Check if this slot is available
                    if action_mask[slot_idx] == 0:
                        continue
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    # Must match commodity
                    if slot_k != sink_k:
                        continue
                    
                    # Check if edge is prohibited
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    edge_head = self.edges['head'][edge_idx]
                    edge_tail = self.edges['tail'][edge_idx]
                    
                    # For now, simple heuristic: send from any source with stock
                    # In a real policy, we'd do proper routing
                    stock_key = (edge_tail, sink_k)
                    if stock_key in self.stock_slot_map:
                        stock_idx = self.stock_slot_map[stock_key]
                        if stock_observed[stock_idx] > 0:
                            available = stock_qty[stock_idx]
                            capacity = graph_capacity[edge_idx]
                            
                            # Reduce capacity by chokepoint closure if applicable
                            for choke_idx, choke_node in enumerate(self.chokepoints):
                                if edge_head == choke_node or edge_tail == choke_node:
                                    capacity *= graph_open[choke_idx]
                            
                            if capacity > 0:
                                flow = min(available, capacity, demand_needed)
                                if flow > 0.01:
                                    flows[slot_idx] = flow
                                    demand_needed -= flow
        except Exception as e:
            # On error, return zeros (naive strategy)
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }