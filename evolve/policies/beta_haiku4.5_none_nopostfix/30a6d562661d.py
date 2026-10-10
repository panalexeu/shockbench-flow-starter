# -2.1086533434634798
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed safely
        seed = config['policy_seed']
        if isinstance(seed, int):
            seed = seed & 0x7FFFFFFF
        else:
            seed = 0
        np.random.seed(seed)
        
        # Extract static structures (dicts of parallel lists)
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.nodes = config['static']['nodes']
        self.commodities = config['static']['commodities']
        self.sinks = config['static']['sinks']
        
        # Extract layout structures
        self.stock_slots = config['layout']['stock_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        
        # Build reverse lookup maps
        self.stock_map = {}
        for i, slot in enumerate(self.stock_slots):
            self.stock_map[tuple(slot)] = i
        
        self.demand_map = {}
        for i, slot in enumerate(self.demands):
            self.demand_map[tuple(slot)] = i
        
        # Extract shortage costs (pi values)
        self.shortage_cost = {}
        for i, (sink_node, sink_k) in enumerate(self.demands):
            for j in range(len(self.sinks['node'])):
                if self.sinks['node'][j] == sink_node and self.sinks['k'][j] == sink_k:
                    self.shortage_cost[i] = self.sinks['pi'][j]
                    break
            if i not in self.shortage_cost:
                self.shortage_cost[i] = 0
    
    def act(self, observation):
        week = int(observation['week'][0])
        
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
            backlog_qty = observation['backlog.qty']
            demand_forecast = observation['demand_forecast.qty']
            graph_u = observation['graph_now.u']
            graph_tau = observation['graph_now.tau']
            graph_prohibited = observation['graph_now.prohibited']
            graph_open = observation['graph_now.open']
            action_mask = observation['action_mask']
            slot_mask = observation['slot_mask']
            pipeline_arrival = observation['pipeline.arrival_week']
            
            # Get demand forecast (multiple horizons)
            horizon = min(8, demand_forecast.shape[1] if demand_forecast.ndim > 1 else 1)
            
            # Calculate total demand per commodity per sink over next few weeks
            # Prioritize by shortage cost (higher cost = higher priority)
            demand_priorities = []
            
            for d_idx in range(len(self.demands)):
                sink_node, sink_k = self.demands[d_idx]
                
                # Sum demand over next few weeks (lookahead)
                total_future_demand = 0
                for h in range(horizon):
                    if demand_forecast.ndim > 1 and h < demand_forecast.shape[1]:
                        d_h = demand_forecast[d_idx, h]
                    else:
                        d_h = demand_forecast[d_idx] if h == 0 else 0
                    
                    if np.isfinite(d_h) and d_h > 0:
                        total_future_demand += d_h
                
                # Account for backlog
                current_backlog = backlog_qty[d_idx] if np.isfinite(backlog_qty[d_idx]) else 0
                net_demand = max(0, total_future_demand - current_backlog)
                
                if net_demand > 0.01:
                    shortage_cost = self.shortage_cost.get(d_idx, 0)
                    priority = (-shortage_cost, d_idx, net_demand, sink_node, sink_k)
                    demand_priorities.append(priority)
            
            # Sort by priority (highest shortage cost first)
            demand_priorities.sort()
            
            # Track which stocks have been allocated to avoid double-counting
            stock_allocated = {}
            edge_flow_used = {}
            
            # Greedy allocation
            for _, d_idx, total_needed, sink_node, sink_k in demand_priorities:
                if total_needed <= 0.01:
                    continue
                
                remaining_needed = total_needed
                
                # Try to find paths that deliver this commodity
                for slot_idx in range(num_action_slots):
                    if remaining_needed <= 0.01:
                        break
                    
                    # Check slot validity
                    if action_mask[slot_idx] == 0:
                        continue
                    if slot_mask[slot_idx] > 0:
                        continue
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    if slot_k != sink_k:
                        continue
                    
                    # Check prohibition
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    edge_tail = self.edges['tail'][edge_idx]
                    edge_head = self.edges['head'][edge_idx]
                    
                    # Check stock at source
                    stock_key = (edge_tail, sink_k)
                    if stock_key not in self.stock_map:
                        continue
                    
                    stock_idx = self.stock_map[stock_key]
                    if stock_observed[stock_idx] == 0:
                        continue
                    
                    available = stock_qty[stock_idx]
                    already_allocated = stock_allocated.get(stock_idx, 0)
                    usable = max(0, available - already_allocated)
                    
                    if usable <= 0.01:
                        continue
                    
                    # Get capacity
                    capacity = float(graph_u[edge_idx])
                    edge_flow_used_now = edge_flow_used.get(edge_idx, 0)
                    capacity = max(0, capacity - edge_flow_used_now)
                    
                    if not np.isfinite(capacity) or capacity <= 0:
                        continue
                    
                    # Apply chokepoint openness
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if choke_idx < len(graph_open):
                                openness = float(graph_open[choke_idx])
                                if np.isfinite(openness):
                                    capacity *= openness
                    
                    if capacity <= 0.01:
                        continue
                    
                    # Allocate flow
                    flow = min(usable, capacity, remaining_needed)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining_needed -= flow
                        
                        # Update tracking
                        stock_allocated[stock_idx] = already_allocated + flow
                        edge_flow_used[edge_idx] = edge_flow_used_now + flow
        
        except Exception:
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }