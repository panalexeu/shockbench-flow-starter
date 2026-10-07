ShockBench-Flow is a weekly control task on a supply network. Every week the
policy decides how much of each good to send along each route. Disruptions (a
strait closes, a route is sanctioned, a tariff jumps, a factory goes down) are
drawn before the episode starts and nothing the policy does changes them. The
policy sees the network as it is this week, its stock and shipments, a demand
forecast, and noisy early warnings and announcements. An episode costs money
(USD); lower is better.

The score (RSS) compares the policy's cost with two references on the same
scenarios: **0** is the naive rule, **1** is the clairvoyant plan (a linear
program that knew every disruption in advance); below 0 is worse than naive and
is not clipped.

## The interface

```python
class Agent:
    def __init__(self, config):    # once per episode
        ...

    def act(self, observation):    # once per week
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
```

- `config`: a dict with keys `static` (the tables below), `layout` (the tables
  below), `T` (weeks), `regime` (the signals' parameters), `policy_seed` (seed
  for random generators), `release_modes` and `spaces` (each observation and
  action field's `shape` and `dtype`).
- `observation`: a dict of numpy arrays keyed by the field names below.
- `flows`: a float array, one quantity (0 or more) per action slot.
- `override_qty`: a float array, one per override slot; its length is
  `len(config["static"]["override_slots"]["chokepoint"])`.
- `release_mode`: an int array, one per `config["layout"]["release_pairs"]`
  entry: 0 default release, 1 release your `override_qty`, 2 hold.
- `override_qty` and `release_mode` may be left out (zeros: the default
  release). A wrong length makes the whole action malformed.
- Entries on a prohibited slot, and negative or non-finite quantities, are
  dropped; the rest of the action stands.

## Field reference (generated from Small)

The fields below, with their meaning. Shapes are shown as on Small to give a
sense of size; on other variants they differ: read them from `config["spaces"]`.

**Observation** (`act(observation)`): a dict of numpy arrays. Every key below except `action_mask.observed` and `override_mask.observed` is followed by `<key>.observed`, an int8 array of the same shape, 1 where the value is present and 0 where it is unobserved or padding (the value is then 0).

| key | shape | dtype | indexed by | meaning |
| --- | --- | --- | --- | --- |
| `week` | (1,) | int64 | - | t, the week to decide (1-based); every other field is the state at instant t - 1 |
| `stock.qty` | (59,) | float64 | layout.stock_slots | on-hand stock I^{t-1} per (node, k), chokepoints excluded (see queue_lots) |
| `backlog.qty` | (8,) | float64 | layout.demands | unserved demand carried at each backlog sink (0 at lost-sales sinks) |
| `pipeline.edge` | (269,) | int64 | grouped list | shipments in transit, one entry per (edge, k, lane, arrival week): the edge |
| `pipeline.k` | (269,) | int64 | grouped list | commodity index |
| `pipeline.lane` | (269,) | int64 | grouped list | lane index; unobserved for shipments off any lane |
| `pipeline.qty` | (269,) | float64 | grouped list | the group's total quantity; its observed-mask marks the live entries |
| `pipeline.arrival_week` | (269,) | int64 | grouped list | week the group reaches the edge's head |
| `queue_lots.qty` | (77, 52) | float64 | layout.lot_keys x week | quantity waiting at a chokepoint, one row per (chokepoint, k, lane, next edge) of layout.lot_keys, one column per week the lots reached it (column w - 1 is week w; their FIFO cohort): the total of those lots, observed where lots wait |
| `wip.node` | (54,) | int64 | padded list | work in process at fabs and OSATs (gross): node |
| `wip.k` | (54,) | int64 | padded list | output commodity |
| `wip.qty` | (54,) | float64 | padded list | quantity |
| `wip.out_week` | (54,) | int64 | padded list | week it becomes stock |
| `graph_now.u` | (115,) | float64 | edges | capacity u per edge this week; unobserved on grid couplings |
| `graph_now.c` | (115,) | float64 | edges | freight cost per unit per edge |
| `graph_now.tau` | (115,) | int64 | edges | lead time in weeks per edge |
| `graph_now.prohibited` | (115, 8) | int8 | edges x commodities | 1 where (edge, k) is prohibited (sanctions, export controls) |
| `graph_now.tariff` | (115, 8) | float64 | edges x commodities | tariff rate on (edge, k) |
| `graph_now.open` | (7,) | float64 | layout.chokepoints | open fraction o_c of each chokepoint (a strait, canal or the Cape route; 1 open, 0 closed) |
| `graph_now.kappa.tb` | (7,) | float64 | layout.chokepoints | throughput kappa_cb of the tanker/bulk pool |
| `graph_now.kappa.ct` | (7,) | float64 | layout.chokepoints | throughput kappa_cb of the container pool |
| `graph_now.war_risk` | (7,) | int64 | layout.chokepoints | war-risk class code: 0 none, 1 red_sea, 2 hormuz_2026 |
| `graph_now.supply.avail` | (10,) | float64 | layout.supply_slots | supply available at each source and material slot |
| `graph_now.fab.R` | (6,) | float64 | layout.fabs | restoration factor R_f of each fab |
| `graph_now.fab.alpha_bar` | (6,) | float64 | layout.fabs | power multiplier alpha-bar_f of each fab |
| `graph_now.fab.cap_eff` | (6,) | float64 | layout.fabs | effective wafer capacity of each fab |
| `graph_now.grid.G_bar` | (4,) | float64 | layout.grids | deliverable generation G-bar_g of each grid |
| `graph_now.grid.y_bar` | (4,) | float64 | layout.grids | base load y-bar_g of each grid |
| `graph_now.osat.R` | (3,) | float64 | layout.osats | restoration factor R^osat of each OSAT |
| `graph_now.osat.thr_eff` | (3,) | float64 | layout.osats | effective throughput thr R^osat of each OSAT |
| `slot_mask` | (108,) | int8 | action slots | 1 where an edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity this week (the wire's convention; see action_mask) |
| `last_week.clip.requested` | (108,) | float64 | action slots | flow you requested last week |
| `last_week.clip.executed` | (108,) | float64 | action slots | flow executed after the capacity clip |
| `last_week.cost_components` | (8,) | float64 | layout.cost_components | last week's cost by component, USD |
| `last_week.sinks.demand` | (8,) | float64 | layout.demands | last week's demand |
| `last_week.sinks.served` | (8,) | float64 | layout.demands | last week's demand served |
| `last_week.sinks.lost` | (8,) | float64 | layout.demands | last week's demand lost |
| `last_week.shed.qty` | (4,) | float64 | layout.grids | power shed at each grid last week |
| `demand_forecast.qty` | (8, 8) | float64 | layout.demands x h | demand forecast for weeks t + h, h = 0..7 |
| `warning.score` | (22,) | float64 | layout.warning_units | early-warning score S^t per region, dyad and chokepoint |
| `messages.msg_id` | (2432,) | int64 | padded list | live announcement threads (announced, not effective, not withdrawn): thread id |
| `messages.channel` | (2432,) | int64 | padded list | channel code: 0 tariff_formal, 1 tariff_informal, 2 tariff_final, 3 sanction_legal, 4 ties_threat, 5 mid_threat |
| `messages.kind` | (2432,) | int64 | padded list | message kind code: 0 proposal, 1 final_notice, 2 threat, 3 publication, 4 withdrawal |
| `messages.region` | (2432,) | int64 | padded list | region index |
| `messages.target_kind` | (2432,) | int64 | padded list | target kind code: 0 chokepoint, 1 edge, 2 node, 3 region |
| `messages.target` | (2432,) | int64 | padded list | target index |
| `messages.k` | (2432,) | int64 | padded list | commodity index; unobserved when the message names none |
| `messages.announced_week` | (2432,) | int64 | padded list | week announced |
| `messages.stated_effective_week` | (2432,) | int64 | padded list | stated effective week; unobserved when none is stated |
| `pending_prohibitions.edge` | (4736,) | int64 | padded list | announced prohibitions not yet in force: edge |
| `pending_prohibitions.k` | (4736,) | int64 | padded list | commodity index |
| `pending_prohibitions.effective_week` | (4736,) | int64 | padded list | week it takes effect |
| `closure_end.chokepoint` | (128,) | int64 | padded list | closures acting now: chokepoint |
| `closure_end.end_week` | (128,) | int64 | padded list | announced end week; unobserved when unknown |
| `action_mask` | (108,) | int8 | action slots | 1 where no edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity this week (the inverse of slot_mask); capacities and closures are not checked: read graph_now.u and graph_now.open |
| `action_mask.observed` | (1,) | int8 | - | 1 when this week's mask was observed (0 in a blackout week: all slots allowed) |
| `override_mask` | (36,) | int8 | override slots | 1 where the slot's own out edge is not prohibited for its commodity this week; release_mode 1 sends every override slot of its pair, and a slot at 0 is dropped whatever its override_qty (one invalid entry, no cost): the pair's other slots stand, and with none valid the default release stays on |
| `override_mask.observed` | (1,) | int8 | - | 1 when this week's override mask was observed (0 in a blackout week) |

**Action** (the return value of `act`): a dict of numpy arrays (`override_qty` and `release_mode` may be left out: zeros, the default release).

| key | shape | dtype | indexed by | meaning |
| --- | --- | --- | --- | --- |
| `flows` | (108,) | float64 | action slots | quantity to dispatch on each (edge, commodity, lane) slot; 0 sends nothing |
| `override_qty` | (36,) | float64 | override slots | tanker cargo to release on each override slot, read where release_mode is 1 |
| `release_mode` | (14,) | int64 | layout.release_pairs | per (chokepoint, tanker commodity): 0 default release, 1 override, 2 hold |

**Layout tables** (`config["layout"]`): each is a list; entry `i` says what row
`i` of the matching dense block stands for. Entries are indices (or names where
stated), never positions to hard-code.

| table             | entry form                                              | indexes                                                                    |
| ----------------- | ------------------------------------------------------- | -------------------------------------------------------------------------- |
| `stock_slots`     | `[node, commodity]`                                     | `stock.qty`                                                                |
| `supply_slots`    | `[node, commodity]`                                     | `graph_now.supply.avail`                                                   |
| `demands`         | `[sink node, commodity]`                                | `backlog.qty`, `demand_forecast.qty` rows, `last_week.sinks.*`             |
| `chokepoints`     | `node`                                                  | `graph_now.open`, `graph_now.kappa.*`, `graph_now.war_risk`                |
| `fabs`            | `node`                                                  | `graph_now.fab.*`                                                          |
| `grids`           | `node`                                                  | `graph_now.grid.*`, `last_week.shed.qty`                                   |
| `osats`           | `node`                                                  | `graph_now.osat.*`                                                         |
| `warning_units`   | `[kind, id]`: kind `"region"` (region index), `"dyad"` (row of `static["dyads"]`) or `"chokepoint"` (node index) | `warning.score` |
| `cost_components` | a name (`"freight"`, `"shortage"`, ...)                 | `last_week.cost_components`                                                |
| `release_pairs`   | `[chokepoint node, commodity]`                          | `release_mode`                                                             |
| `lot_keys`        | `[chokepoint node, commodity, lane, next edge]` (Small and Full only) | `queue_lots.qty` rows                                        |

Action slots (`flows`, `action_mask`, `slot_mask`, `last_week.clip.*`) are
indexed by `static["action_slots"]` (`edge`, `k`, `lane`); override slots
(`override_qty`, `override_mask`) by `static["override_slots"]` (`chokepoint`,
`k`, `out_edge`, `lane`).

**Static tables** (`config['static']`, the episode's public tables): a table is a dict of equal-length lists, one entry per node, edge, lane, commodity, slot or sink, and the indices above point into them.

| field | on Small | meaning |
| --- | --- | --- |
| `instance` | keys `T`, `chokepoint_adjacency`, `commodities`, `compatibility`, `edges`, `initial_state`, `instance_id`, `kind`, `lanes`, `nodes`, `params`, `prohibitions_at_reset`, `provenance`, `region_class`, `regions`, `routing_table`, `schema_version`, `trade_adjacency`, `units`, `use` | the full public instance JSON; `initial_state.pipeline` holds the week-0 shipments of the nominal plan; edges there call the lead time `tau` |
| `instance_id` | `chokepoint-small` | the instance's name |
| `instance_hash` | `df1f84afdbc1...` | SHA-256 of the instance |
| `T` | `52` | the horizon, in weeks |
| `units` | chip_le: wafer-eq 300 mm, chip_le_raw: wafer-eq 300 mm, chip_mat: wafer-eq 300 mm, chip_mat_raw: wafer-eq 300 mm, cost: USD, crude: GWh fuel, lng: GWh fuel, nucfuel: GWh fuel, wafer: wafer-eq 300 mm | the unit of each commodity's quantities, and of costs |
| `regions` | 14 entries | region names: `nodes.region`, `messages.region`, `dyads.*` and the warning's region units index them |
| `nodes.id` | 38 entries | node name |
| `nodes.type` | 38 entries | source, terminal, grid, chokepoint (a strait, canal or the Cape route), material, fab, osat or sink |
| `nodes.region` | 38 entries | region index |
| `commodities.id` | 8 entries | commodity name |
| `commodities.v` | 8 entries | customs value v_k, USD per unit |
| `commodities.pool` | 8 entries | chokepoint throughput pool: tb (tanker/bulk, `graph_now.kappa.tb`) or ct (container) |
| `commodities.override` | 8 entries | true for a tanker commodity, whose cargo queued at a chokepoint `release_mode` and `override_qty` steer |
| `edges.id` | 115 entries | edge name |
| `edges.tail` | 115 entries | node index the edge leaves |
| `edges.head` | 115 entries | node index the edge reaches |
| `edges.mode` | 115 entries | sea, air, pipeline or grid |
| `edges.tau0` | 115 entries | nominal lead time in weeks (`graph_now.tau` is this week's) |
| `edges.c0` | 115 entries | nominal freight, USD per unit (`graph_now.c` is this week's) |
| `edges.u0` | 115 entries | nominal capacity per week, null on a grid coupling (`graph_now.u` is this week's) |
| `edges.K` | 115 entries | commodity indices the edge may carry (empty on a grid coupling) |
| `edges.alt_of` | 115 entries | the route the edge duplicates, {'lane': i} or {'edge': i}, or null |
| `edges.pool` | 115 entries | chokepoint pool of its traffic, null on a grid coupling |
| `lanes.id` | 39 entries | lane name (a route through one or more chokepoints) |
| `lanes.edges` | 39 entries | edge indices along the lane, in order |
| `lanes.chokepoints` | 39 entries | chokepoint node indices it passes, in order |
| `lanes.alt_of` | 39 entries | the route the lane duplicates, {'lane': i} or {'edge': i}, or null |
| `action_slots.edge` | 108 entries | each slot's edge (on a lane, the lane's first edge) |
| `action_slots.k` | 108 entries | each slot's commodity index |
| `action_slots.lane` | 108 entries | each slot's lane index, null off any lane |
| `override_slots.chokepoint` | 36 entries | chokepoint node index |
| `override_slots.k` | 36 entries | tanker commodity index |
| `override_slots.out_edge` | 36 entries | edge the released cargo leaves the chokepoint by |
| `override_slots.lane` | 36 entries | lane index, null off any lane |
| `sinks.node` | 8 entries | demand node index (the rows of `layout.demands`) |
| `sinks.k` | 8 entries | commodity demanded |
| `sinks.backlog` | 8 entries | true: unserved demand is carried to later weeks; false: it is lost |
| `sinks.pi` | 8 entries | shortage penalty pi, USD per unit of demand not served |
| `dyads.a` | 1 entry | first region index of each dyad (the warning's dyad units) |
| `dyads.b` | 1 entry | second region index of each dyad |
| `regime` | `standard` | the information regime's published parameters (`name`, `L`, `a`, `phi`, `chi`, `h_cov`, `skill`, `blackout`); null when the runner hides it |

---