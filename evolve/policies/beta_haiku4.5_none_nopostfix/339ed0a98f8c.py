# -2.109105800307275
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed safely
        seed = config['policy_seed']
        if isinstance(seed, int):
            np.random.seed(seed & ((1 << 31) - 1))
        else:
            np.random.seed(0)
        
        # Static structures: dicts of parallel lists
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.commodities = config['static']['commodities']
        self.sinks = config['static']['sinks']
        
        # Layout structures: lists of index tuples
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
        
        # Build shortage penalty map from sinks
        self.pi_by_demand = {}
        for i, (sink_node, sink_k) in enumerate(self.demands):
            pi = 0.0
            for j in range(len(self.sinks['node'])):
                if self.sinks['node'][j] == sink_node and self.sinks['k'][j] == sink_k:
                    pi = float(self.sinks['pi'][j])
                    break
            self.pi_by_demand[i] = pi
    
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
            stock_obs = observation['stock.qty.observed']
            backlog_qty = observation['backlog.qty']
            demand_forecast = observation['demand_forecast.qty']
            graph_u = observation['graph_now.u']
            graph_tau = observation['graph_now.tau']
            graph_prohibited = observation['graph_now.prohibited']
            graph_open = observation['graph_now.open']
            graph_c = observation['graph_now.c']
            graph_tariff = observation['graph_now.tariff']
            action_mask = observation['action_mask']
            slot_mask = observation['slot_mask']
            pipeline_qty = observation['pipeline.qty']
            pipeline_arrival = observation['pipeline.arrival_week']
            
            # Extract immediate demand (week t, column 0)
            if demand_forecast.ndim > 1:
                demand_t = demand_forecast[:, 0]
                # Also look at week t+1 for planning
                demand_t1 = demand_forecast[:, 1] if demand_forecast.shape[1] > 1 else np.zeros(len(self.demands))
            else:
                demand_t = demand_forecast
                demand_t1 = np.zeros(len(self.demands))
            
            # Build list of demands with priorities (highest shortage cost first)
            demand_priorities = []
            for d_idx in range(len(self.demands)):
                sink_node, sink_k = self.demands[d_idx]
                
                d_t = float(demand_t[d_idx]) if np.isfinite(demand_t[d_idx]) else 0.0
                d_t1 = float(demand_t1[d_idx]) if np.isfinite(demand_t1[d_idx]) else 0.0
                backlog = float(backlog_qty[d_idx]) if np.isfinite(backlog_qty[d_idx]) else 0.0
                
                # Total needed over next 2 weeks minus backlog
                total_need = max(0.0, d_t + d_t1 - backlog)
                
                if total_need > 0.01:
                    pi = self.pi_by_demand.get(d_idx, 0.0)
                    # Priority: higher shortage cost = higher priority (negate for sorting)
                    demand_priorities.append((-pi, d_idx, total_need, sink_node, sink_k))
            
            # Sort by priority (most critical first)
            demand_priorities.sort()
            
            # Track stock and edge capacity consumed
            stock_consumed = {}
            edge_flow = {}
            
            # Allocate flows greedily by demand priority
            for _, d_idx, need, sink_node, sink_k in demand_priorities:
                if need <= 0.01:
                    continue
                
                remaining = need
                
                # Try each action slot to satisfy this demand
                for slot_idx in range(num_action_slots):
                    if remaining <= 0.01:
                        break
                    
                    # Check slot validity
                    if action_mask[slot_idx] == 0 or slot_mask[slot_idx] > 0:
                        continue
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    # Commodity must match
                    if slot_k != sink_k:
                        continue
                    
                    # Check prohibition
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    edge_tail = self.edges['tail'][edge_idx]
                    edge_head = self.edges['head'][edge_idx]
                    
                    # Check if stock is available at source
                    stock_key = (edge_tail, sink_k)
                    if stock_key not in self.stock_map:
                        continue
                    
                    stock_idx = self.stock_map[stock_key]
                    if stock_obs[stock_idx] == 0:
                        continue
                    
                    available_stock = float(stock_qty[stock_idx])
                    already_consumed = stock_consumed.get(stock_idx, 0.0)
                    usable_stock = max(0.0, available_stock - already_consumed)
                    
                    if usable_stock <= 0.01:
                        continue
                    
                    # Get edge capacity
                    capacity = float(graph_u[edge_idx])
                    if not np.isfinite(capacity) or capacity <= 0:
                        continue
                    
                    edge_used = edge_flow.get(edge_idx, 0.0)
                    free_capacity = max(0.0, capacity - edge_used)
                    
                    # Apply chokepoint openness
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if choke_idx < len(graph_open):
                                openness = float(graph_open[choke_idx])
                                if np.isfinite(openness):
                                    free_capacity *= openness
                    
                    if free_capacity <= 0.01:
                        continue
                    
                    # Allocate flow
                    flow = min(usable_stock, free_capacity, remaining)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining -= flow
                        stock_consumed[stock_idx] = already_consumed + flow
                        edge_flow[edge_idx] = edge_used + flow
        
        except Exception:
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }