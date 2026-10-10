# -2.109105800307275
import numpy as np

class Agent:
    def __init__(self, config):
        self.config = config
        self.T = config['T']
        
        # Seed safely - handle large seeds
        seed = config['policy_seed']
        if isinstance(seed, int):
            # Use modulo to fit into valid range
            seed = (seed % (2**31)) if seed >= 0 else ((-seed) % (2**31))
        else:
            seed = 0
        np.random.seed(seed)
        
        # Static structures
        self.action_slots = config['static']['action_slots']
        self.override_slots = config['static']['override_slots']
        self.edges = config['static']['edges']
        self.commodities = config['static']['commodities']
        self.sinks = config['static']['sinks']
        
        # Layout structures
        self.stock_slots = config['layout']['stock_slots']
        self.demands = config['layout']['demands']
        self.chokepoints = config['layout']['chokepoints']
        self.release_pairs = config['layout']['release_pairs']
        
        # Build maps
        self.stock_map = {tuple(slot): i for i, slot in enumerate(self.stock_slots)}
        self.demand_map = {tuple(d): i for i, d in enumerate(self.demands)}
        
        # Build shortage cost map
        self.pi_map = {}
        for i, (sink_node, sink_k) in enumerate(self.demands):
            for j in range(len(self.sinks['node'])):
                if self.sinks['node'][j] == sink_node and self.sinks['k'][j] == sink_k:
                    self.pi_map[i] = float(self.sinks['pi'][j])
                    break
            if i not in self.pi_map:
                self.pi_map[i] = 0.0
    
    def act(self, observation):
        week = int(observation['week'][0])
        
        num_action_slots = len(self.action_slots['edge'])
        num_override_slots = len(self.override_slots['chokepoint'])
        num_release_pairs = len(self.release_pairs)
        
        flows = np.zeros(num_action_slots, dtype=np.float64)
        override_qty = np.zeros(num_override_slots, dtype=np.float64)
        release_mode = np.zeros(num_release_pairs, dtype=np.int64)
        
        try:
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
            last_week_clip_executed = observation.get('last_week.clip.executed')
            
            # Get demand - use week 0 (current week)
            if demand_forecast.ndim > 1:
                demand_this_week = demand_forecast[:, 0]
                demand_next_week = demand_forecast[:, 1] if demand_forecast.shape[1] > 1 else np.zeros(len(self.demands))
            else:
                demand_this_week = demand_forecast
                demand_next_week = np.zeros(len(self.demands))
            
            # Build demand list with priorities
            demand_list = []
            for d_idx in range(len(self.demands)):
                sink_node, sink_k = self.demands[d_idx]
                
                d0 = float(demand_this_week[d_idx]) if np.isfinite(demand_this_week[d_idx]) else 0.0
                d1 = float(demand_next_week[d_idx]) if np.isfinite(demand_next_week[d_idx]) else 0.0
                backlog = float(backlog_qty[d_idx]) if np.isfinite(backlog_qty[d_idx]) else 0.0
                
                # Total urgent need (this + next week, minus backlog)
                urgent_need = max(0, d0 + d1 - backlog)
                
                if urgent_need > 0.01:
                    # Priority by shortage cost (higher = more critical)
                    pi = self.pi_map.get(d_idx, 0.0)
                    demand_list.append((-pi, d_idx, urgent_need, sink_node, sink_k))
            
            # Sort by priority
            demand_list.sort()
            
            # Track cumulative allocation per stock
            stock_alloc = {}
            edge_alloc = {}
            
            # For each high-priority demand, allocate flows
            for _, d_idx, needed, sink_node, sink_k in demand_list:
                if needed <= 0.01:
                    continue
                
                remaining = needed
                
                # Try all slots that could deliver this commodity
                for slot_idx in range(num_action_slots):
                    if remaining <= 0.01:
                        break
                    
                    if action_mask[slot_idx] == 0 or slot_mask[slot_idx] > 0:
                        continue
                    
                    edge_idx = self.action_slots['edge'][slot_idx]
                    slot_k = self.action_slots['k'][slot_idx]
                    
                    if slot_k != sink_k:
                        continue
                    
                    if graph_prohibited[edge_idx, sink_k] > 0:
                        continue
                    
                    edge_tail = self.edges['tail'][edge_idx]
                    edge_head = self.edges['head'][edge_idx]
                    
                    # Check stock
                    stock_key = (edge_tail, sink_k)
                    if stock_key not in self.stock_map:
                        continue
                    
                    s_idx = self.stock_map[stock_key]
                    if stock_observed[s_idx] == 0:
                        continue
                    
                    avail = float(stock_qty[s_idx])
                    allocated = stock_alloc.get(s_idx, 0.0)
                    usable = max(0.0, avail - allocated)
                    
                    if usable <= 0.01:
                        continue
                    
                    # Check edge capacity
                    cap = float(graph_u[edge_idx])
                    if not np.isfinite(cap):
                        cap = 0.0
                    
                    edge_used = edge_alloc.get(edge_idx, 0.0)
                    cap = max(0.0, cap - edge_used)
                    
                    if cap <= 0.01:
                        continue
                    
                    # Apply chokepoint openness
                    for choke_idx, choke_node in enumerate(self.chokepoints):
                        if edge_head == choke_node or edge_tail == choke_node:
                            if choke_idx < len(graph_open):
                                openness = float(graph_open[choke_idx])
                                if np.isfinite(openness):
                                    cap *= openness
                    
                    if cap <= 0.01:
                        continue
                    
                    # Allocate
                    flow = min(usable, cap, remaining)
                    if flow > 0.01:
                        flows[slot_idx] = flow
                        remaining -= flow
                        stock_alloc[s_idx] = allocated + flow
                        edge_alloc[edge_idx] = edge_used + flow
        
        except Exception:
            pass
        
        return {
            'flows': flows,
            'override_qty': override_qty,
            'release_mode': release_mode
        }