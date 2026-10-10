# -999
# error: index 14 is out of bounds for axis 0 with size 14
import numpy as np

class Agent:
    """Aggressive backlog-driven with pipeline depletion and tanker releases."""
    
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
        self.override_slots = static["override_slots"]
        self.lot_keys = config["layout"].get("lot_keys", [])
        
        self.override_qty = np.zeros(action_space["override_qty"]["shape"], dtype=float)
        self.release_mode = np.zeros(action_space["release_mode"]["shape"], dtype=np.int64)
    
    def act(self, observation):
        flows = self.capacity.copy()
        action_mask = observation["action_mask"]
        flows = flows * action_mask
        
        # Compute backlog and pipeline per commodity
        backlog = observation["backlog.qty"]
        pipeline_qty = observation["pipeline.qty"]
        pipeline_k = observation["pipeline.k"]
        pipeline_obs = observation["pipeline.qty.observed"]
        
        num_commodities = len(self.commodities["id"])
        backlog_per_k = np.zeros(num_commodities)
        pipeline_per_k = np.zeros(num_commodities)
        
        # Aggregate backlog by commodity
        for sink_idx in range(min(len(backlog), len(self.sinks["k"]))):
            k_idx = self.sinks["k"][sink_idx]
            backlog_per_k[k_idx] += backlog[sink_idx]
        
        # Aggregate pipeline by commodity
        obs_indices = np.where(pipeline_obs)[0]
        for i in obs_indices:
            k_idx = int(pipeline_k[i])
            if 0 <= k_idx < num_commodities:
                pipeline_per_k[k_idx] += pipeline_qty[i]
        
        # Compute normalized pressure signals
        max_backlog = np.max(backlog_per_k) if backlog_per_k.max() > 0 else 1.0
        max_pipeline = np.max(pipeline_per_k) if pipeline_per_k.max() > 0 else 1.0
        
        backlog_pressure = backlog_per_k / (max_backlog + 1e-9)
        pipeline_pressure = 1.0 - (pipeline_per_k / (max_pipeline + 1e-9))
        
        # Combined pressure: more aggressive weighting on backlog
        combined_pressure = 0.8 * backlog_pressure + 0.2 * pipeline_pressure
        
        # Apply aggressive boost: up to 1.30x
        for slot_idx in range(len(flows)):
            if action_mask[slot_idx] > 0:
                k_idx = self.action_slots["k"][slot_idx]
                boost_factor = 1.0 + 0.30 * combined_pressure[k_idx]
                flows[slot_idx] *= boost_factor
        
        # Soft cap to avoid excessive over-allocation
        allowed_capacity = self.capacity * action_mask
        total_allowed = np.sum(allowed_capacity)
        total_flows = np.sum(flows)
        if total_flows > total_allowed * 1.25:
            flows = flows * (total_allowed * 1.2 / (total_flows + 1e-9))
        
        # Tanker release strategy: release more when queue lots are deep
        release_mode_out = self.release_mode.copy()
        override_qty_out = self.override_qty.copy()
        
        if "queue_lots.qty" in observation:
            queue_lots = observation["queue_lots.qty"]
            queue_obs = observation["queue_lots.qty.observed"]
            
            # For each override slot, check if its chokepoint has deep queues
            for override_idx in range(len(self.override_slots["chokepoint"])):
                choke_node = self.override_slots["chokepoint"][override_idx]
                override_k = self.override_slots["k"][override_idx]
                
                # Find lot_keys rows for this chokepoint
                total_queue_depth = 0.0
                if len(self.lot_keys) > 0:
                    for lot_idx in range(min(len(self.lot_keys), queue_lots.shape[0])):
                        lot_choke = self.lot_keys[lot_idx][0]
                        lot_k = self.lot_keys[lot_idx][1]
                        if lot_choke == choke_node and lot_k == override_k:
                            # Sum across arrival weeks
                            total_queue_depth += np.sum(queue_lots[lot_idx])
                
                # If queue is deep, trigger override release
                if total_queue_depth > 0.1:  # Threshold for releasing tanker cargo
                    release_mode_out[override_idx] = 1  # Override mode
                    # Release proportional to queue depth, capped at capacity
                    override_qty_out[override_idx] = min(total_queue_depth * 0.5, self.capacity[0] * 10)
        
        return {
            "flows": flows,
            "override_qty": override_qty_out,
            "release_mode": release_mode_out
        }
