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

## How the network works

The network couples two supply chains:
- **Energy**: `source` nodes (LNG, crude, nuclear fuel) ship by sea, through `chokepoint` nodes, to `terminal`
  nodes, which feed `grid` nodes. A grid burns fuel to serve its base load and to power fabs.
- **Chips**: `material` nodes (wafers) ship to `fab` nodes, which turn wafers into raw chips; raw chips ship to
  `osat` nodes (packaging plants), which turn them into packaged chips; packaged chips ship to `sink` nodes,
  where the demand is.
- Grids power fabs through edges of mode `grid` (coupling edges, no goods, `u0` null). A fab whose grid is
  short of fuel starts fewer wafers.

Goods move only along action slots (`flows`) and, at chokepoints, by queue release. Production, packaging, grid
dispatch and serving demand are done by the simulator, not by the policy.

## One week, in order

After `act` returns, the simulator runs week t:
1. **Chokepoint release.** Cargo waiting at each chokepoint is released onto its next edge, oldest arrivals
   first. Per pool, at most the throughput `graph_now.kappa.*` (= k_c x mu x open fraction) is released, and no
   more than the next edge's capacity. Cargo whose next edge is prohibited waits. For tanker commodities,
   `release_mode` 1 releases `override_qty` on the named slots instead, and 2 holds everything.
2. **Clip.** Requests on prohibited slots are dropped. The requests starting on one edge are scaled down
   together to its capacity `graph_now.u`; the requests drawing on one (node, k) stock are scaled down together
   to that stock; and the shipping fleet of each pool is shared pro rata. `last_week.clip.executed` shows what
   went through.
3. **Dispatch.** Executed flows leave the stock held at the end of last week (`stock.qty`). A shipment takes
   tau weeks per edge; on a lane it waits in the queue of each chokepoint it reaches.
4. **Arrive.** Shipments due this week join the stock at their edge's head.
5. **Produce.**
   - Sources and material nodes refill up to `storage` from this week's supply (`graph_now.supply.avail`).
   - A grid's output per fuel is shares[k] x `graph_now.grid.G_bar`, limited by the fuel in stock. The rationed
     fuel is cut further when last week's stock is below psi x ibar. The output serves base load and the fabs'
     energy needs in the grid's `priority` order; unserved base load is shed. Under `base_first` (every grid
     on Small) base load is served before any fab gets energy: fabs get only what is left, so a fab's output
     depends on the fuel its grid has beyond its base load. `proportional` scales base load and fabs by the
     same factor; `industrial_first` serves the fabs first.
   - Each fab starts min(alpha_bar x R x cap0, wafers in stock) wafers, reduced if its grid gives it less energy
     than e per wafer; they become raw chips `tau` weeks later. A fab hit scraps part of the last `w_scr`
     weeks' starts.
   - Each OSAT packages raw chips up to `thr_eff` per week (pro rata across its products), ready `tau` weeks later.
6. **Serve.** Each sink serves min(demand, stock). The rest is lost (`backlog` false) or carried over (true).
7. **Dispose.** Stock above `storage` at a node (not sources or materials) is destroyed.

## Chokepoint queues and tanker releases

- Cargo sent on a lane stops at every chokepoint the lane passes and joins that chokepoint's queue (a lot book,
  `queue_lots.qty`; cargo still at sea towards it is in `pipeline.*`). It leaves only when released.
- The default release (`release_mode` 0) takes lots oldest arrival first and sends each one onto the next edge
  of the lane it was dispatched on. It is limited by the chokepoint's throughput per pool (`graph_now.kappa.*`)
  and by the next edge's capacity, shared pro rata among lots of the same arrival week. A lot whose next edge is
  prohibited for its commodity is not released and keeps waiting, paying queue holding every week.
- For tanker commodities (`commodities.override` true: LNG and crude), each (chokepoint, commodity) pair of
  `layout.release_pairs` also has two other modes for the week:
  - `release_mode` 1 (override): the default release is off for that pair, and what is released is
    `override_qty` on each override slot of the pair. An override slot is (chokepoint, commodity, out edge,
    lane): the released cargo leaves by that out edge and continues on that lane (on no lane for a turn-back
    edge), whatever lane it was dispatched on. The quantity is taken from the pair's queue oldest first and
    scaled down to the out edge's capacity, the queue content and the pool's throughput. A slot whose out edge
    is prohibited (`override_mask` 0) is dropped; with no valid slot the default release stays on.
  - `release_mode` 2 (hold): nothing of that commodity leaves the chokepoint this week.
- Container commodities (wafers and chips) always get the default release.
- On Small, for example, the override slots let LNG queued at chk_hormuz leave towards chk_malacca, chk_suez,
  chk_cape or the Lombok route; LNG queued at chk_suez turn back to term_eu; and crude queued at chk_malacca take
  the east route to term_kr or term_jp. Without overrides, queued cargo only ever continues on the lane it was
  dispatched on.

## Cost

Each week:
- freight: c_e x flow, on every edge, dispatches and chokepoint releases alike
- war_risk: a per-unit charge on flows over edges under a war-risk class
- tariff: tariff rate x v_k x flow, charged in the dispatch week
- holding: holding_cost x end-of-week stock (non-chokepoint nodes)
- queue_holding: per-unit charge on cargo waiting at chokepoints, by war-risk class
- shortage: pi x (lost + backlog) at each sink
- disposal: disposal_cost x stock destroyed above storage
- shed: voll x base load shed at each grid

At week T the episode is credited the salvage value of what remains: stock at its node's `salvage`, shipments in
transit at the salvage of the node they are heading to, and fab and OSAT work in process at the salvage of their
input. Sources and materials have salvage 0.

Magnitudes on Small (USD per unit):
- shortage pi: 50,000 (chip_le), 10,000 (chip_mat)
- shed voll: about 4,100,000 per GWh
- customs value v: about 40,000 (lng, crude), 206,000 (nucfuel), 3,000 (wafer)
- disposal cost: 2,000 to 4,000
- holding cost: 2 to 35 per week
- freight c0: 0 to 4,000

## Where the model's numbers are

`config["static"]["instance"]` names nodes, edges and commodities by id, not index:
- `nodes[i].stock[k]`: `storage` (the cap above which stock is destroyed), `holding_cost`, `salvage`, and
  `supply_rate` at sources and materials
- `nodes[i].fab`: `input`, `product`, `cap0`, `tau`, `e` (energy per wafer), `grid`, `w_scr`
- `nodes[i].osat`: `packages` (raw -> packaged), `tau`, `thr`
- `nodes[i].grid`: `base_load`, `deliverable`, `shares` (per fuel and an unmodelled share), `ibar`, `rationed`,
  `priority`, `voll`
- `nodes[i].chokepoint`: `k_c`, `mu` per pool, `queue_holding` and `war_risk_cost` per commodity and war-risk class
- `nodes[i].terminal.throughput`: the capacity of the terminal's edge to its grid
- `nodes[i].sink.demand[k]`: `dbar` (mean weekly demand), `phi`, `sigma` (demand noise), `pi`, `backlog`
- `commodities`: `v`, `disposal_cost`, `pool`, `override`
- `params.psi` (the rationing threshold)
- `use`: which material feeds which fab; `compatibility`: which source can feed which terminal;
  `routing_table`: the lanes between regions
- `initial_state`: stock, pipeline, fab_wip, osat_wip and queue_lots at week 0, which is the normal plan in motion

## Disruptions

Disruptions come from a fixed random generator, the same for every episode; each episode is one draw. Event
types:
- chokepoint closures (weather or military), with war-risk classes
- sanctions and export controls (prohibitions)
- tariffs
- port strikes (edge capacity)
- piracy (freight surcharges)
- regional conflicts (fab and OSAT capacity)
- energy shocks (grid output)
- material outages (supply)
- fab hits (scrap of work in process)

Events last for random durations, and one event raises the chance of others. Warnings and announcements are
noisy: they can be early, late, wrong or missing.

## Reference source: the simulator

The modules below are the simulator's own code from the `shockbench_flow` package, for reference only: the
package is not installed on the scoring server, so a policy cannot import it. Equation numbers such as (23) and
question ids such as Q58 refer to the package's design notes and can be ignored.

### `shockbench_flow/dynamics/sim.py`

```python
"""The weekly simulator in the order of design §3.2 (V4), on one instance and one ``WeeklyMarks``.

Steps of week t: 3 forward at chokepoints (§3.4 pseudocode: arrivals join the lot book; overrides first; default release
by arrival cohort, FIFO across cohorts, pro rata inside a cohort (10); fleet slack last), 4 clip of the requests (3)-(7)
(mask, joint edge cap, shared stock pro rata over the lane sub-requests of each (edge, commodity), fleet slack per pool
over every release that matches a term of ``Instance.dup_items``, chokepoint releases included: an edge-level sea
duplicate or turn-back counts all its flow, a lane term the flow dispatched on its lane), 5 dispatch (2),
6 arrive, 7 produce (supply lift refilling to the cap (Q79); grid segments with gas rationing on fuel on hand (15)-(18)
under pri_g; fab lots (12) with gross WIP; scrap booked in onset weeks (14); OSAT packaging (19) up to
thr_i R_osat_i(t), the OSAT restoration of (13) under a regional conflict (§4.4)), 8 serve (20),
9 charge (disposal above I^max at non-chokepoint, non-supply nodes; C_t (23) and C^¢_t (24)). Step 10 reads the next
week's marks; no random draw happens anywhere.

Floating-point order: the design's reference code ``scripts/python/evidence/tiny_fixture.py`` (class ``Sim``) fixes the
operation order that reproduces the frozen integer cents of §2.4; this module follows it: dispatch draws each executed
slot from its stock in slot order; shipments are appended to the pipeline as dispatches (slot order) then chokepoint
releases (release order), and arrivals are added one shipment at a time in pipeline order; each lane queue Q_ckl is
recomputed from its lots in book order and the chokepoint stock is their fsum over lanes (``cost.queue_totals``, the
LP's formation, §12 'Cost terms (23)'); supply lifts, grids, fabs and OSATs run in instance order; the rationing
factor reads I^{t-1}, the available gas is ``min((zeta G-bar) ration, fuel on hand after steps 5-6)``, the segment
load factor is ``(sum E + y) / G-av``; holding is charged on end-of-week stock after disposal.

Rules the design leaves open, with the reading chosen here (conservative; design §12 'Phase-3 implementation notes'):

- Stock clamps (Q58 M11): a stock that a dispatch, a grid burn or packaging leaves in [-band, 0) is set to 0 and
  logged in ``StepRecord.clamps`` as (slot, value), with band = 1e-12 max(A-bar, |q|, draw, smallest normal float),
  q the quantity taken; A-bar is what bounded it (I^{t-1} for a dispatch, the segment's G^av for a burn, the raw stock
  for packaging) and the draw what the factor that scaled it was computed from (a dispatch's request capped at u'_e,
  the grid's requested load y-bar + sum E-hat of (18) for a burn, the OSAT throughput thr R_osat for packaging). With
  a subnormal A-bar that factor ((5), the served share of (18), (19)) is subnormal and errs by up to 2^-1075
  absolutely, so the draw, not A-bar, bounds the excess; below the smallest normal float every operation errs by up to
  2^-1074. Anything more negative is a bug and raises ``RuntimeError``; a valid action never does (M1 pre-gate
  ORACLE-PRE-1, INT-1, DET-1).
- Edge clamps (Q95): on an edge whose factor (4) is subnormal, the clip takes the executed requests from u'_e in slot
  order under the same rule (``clip.edge_clamp``): a residual left in [-band, 0), with band = 1e-12 max(u'_e, |x|, q,
  smallest normal float) and q the slot's request, lowers that slot to the residual before it and is logged in
  ``StepRecord.edge_clamps`` as (action slot, value); the subnormal factor errs by up to q 2^-1075, which the band
  bounds. The override step at chokepoints applies the same rule to the override slots on an out-edge whose factor (4)
  is subnormal, before (6) (``chokepoint._overrides``), and logs it apart, in ``StepRecord.override_clamps`` as
  (override slot, value), q the override request. Anything lower raises ``RuntimeError``. A week without an edge clamp
  is unchanged bit for bit, and so is its record's hash.
- Dust lots: every week, after the releases, every queue lot with 0 < qty <= 1e-12 leaves the book, whether or not
  anything was released from it (a held lot, a lot on a prohibited next edge or one that arrived this week alike), as
  the reference does (``tiny_fixture.py`` filters the whole book), and is logged in ``clamps`` with its (positive)
  quantity.
- Scrap (14) is booked for every hit in ``marks.fab_hits`` whose onset week is t, on the gross WIP of start weeks
  [t - w_scr, t - 1] (initial WIP included), before this week's output matures, so (12) holds with the same totals.
  Whether a carried-in event (onset <= 0) is a fab hit is the marks module's decision, not the simulator's.
- A hold and override slots on the same (c, k) in one week: the hold wins (nothing is released).
- Supply lift is ``max(0, min(avail, I^max - stock))``; the guard only matters if a supply node ever receives
  shipments.
- The rationing factor min{1, I^{t-1} / (psi I-bar_g)} of (15), (18) is 1 whenever I^{t-1} >= psi I-bar_g, else
  I^{t-1} / (psi I-bar_g): the LP's row psi I-bar_g G <= zeta G-bar I^{t-1} read as a factor, so psi I-bar_g = 0 (0/0
  included) does not ration, and for psi I-bar_g > 0 the float is the reference's ``min`` (M1 pre-gate DC-1).
"""

from dataclasses import dataclass

import numpy as np

from shockbench_flow.dynamics import chokepoint as chk_mod
from shockbench_flow.dynamics.clip import CLAMP_TOL, FLOAT_MIN, clip_requests, fleet_caps, fleet_slack
from shockbench_flow.dynamics.cost import queue_totals, salvage, weekly_costs
from shockbench_flow.dynamics.production import allocate_energy, package
from shockbench_flow.dynamics.state import Lot, Shipment, State, StepRecord, cents
from shockbench_flow.instance.schema import FabAttrs, Instance
from shockbench_flow.marks import WeeklyMarks, osat_throughput


LOT_EPS = 1e-12  # a queue lot at or below this quantity leaves the book (reference rule)


@dataclass(frozen=True)
class _Tables:
    """Week-invariant tables of one instance for ``step`` (kept in ``Instance.memo``)."""

    fleet_caps: tuple[float, float]  # s^b_fl F^b per pool (7)
    fab_attrs: tuple[FabAttrs, ...]  # per fab ordinal
    chokepoint_slots: tuple[tuple[int, int, int], ...]  # (slot, chokepoint node, k), slot order
    supply_slots: tuple[tuple[int, float], ...]  # (slot, I^max) at supply nodes, slot order
    disposal_slots: tuple[tuple[int, float], ...]  # (slot, I^max) at non-chokepoint, non-supply nodes, slot order

    @staticmethod
    def build(inst: Instance) -> "_Tables":
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        slots = list(enumerate(inst.stock_slots))
        return _Tables(
            fleet_caps=fleet_caps(inst),
            fab_attrs=tuple(inst.nodes[f].fab for f in inst.fabs),
            chokepoint_slots=tuple((s, sl.node, sl.k) for s, sl in slots if sl.node in chk),
            supply_slots=tuple((s, sl.storage) for s, sl in slots if sl.node in supply),
            disposal_slots=tuple((s, sl.storage) for s, sl in slots if sl.node not in chk and sl.node not in supply),
        )


def initial_stock(inst: Instance) -> np.ndarray:
    """I^0 per stock slot (§2.3), shared by the simulator and the oracle LP.

    Repeated initial-stock entries of a slot add up, in entry order. A non-chokepoint slot holds that sum; a chokepoint
    slot holds the fsum over lanes of its lane queues, each the lane's lots added in lot order (``lane_queues``, as at
    the end of every week), and its declared stock, the sum of its entries, must equal it within 1e-9 max(1, declared),
    the loader's check (§12 'Loader checks').

    Raises:
        ValueError: if a declared stock or a queue lot has no stock slot, or a declared chokepoint stock differs from
            its queue lots.

    """
    stock = np.zeros(len(inst.stock_slots))
    chk = set(inst.chokepoints)

    def slot(node: int, k: int) -> int:
        s = inst.slot_index.get((node, k))
        if s is None:
            raise ValueError(f"initial state: node {inst.nodes[node].id} has no stock of {inst.commodities[k].id}")
        return s

    for (c, k), q in queue_totals(lane_queues(inst.initial_state.queue_lots)).items():
        stock[slot(c, k)] = q
    declared: dict[int, float] = {}
    for node, k, qty in inst.initial_state.stock:
        s = slot(node, k)
        if node not in chk:
            stock[s] += qty
        else:
            declared[s] = declared.get(s, 0.0) + qty
    for s, qty in declared.items():
        if abs(stock[s] - qty) > 1e-9 * max(1.0, qty):
            node = inst.nodes[inst.stock_slots[s].node].id
            raise ValueError(f"initial stock at {node} ({qty!r}) differs from its queue lots ({float(stock[s])!r})")
    return stock


def initial_state(inst: Instance) -> State:
    """The declared initial state (§2.3): stock, pipeline, fab and OSAT WIP (gross), queue lots, zero backlog."""
    ist = inst.initial_state
    lots = [
        Lot(i, q.chokepoint, q.k, q.qty, q.lane, q.next_edge, q.dispatch_week, q.entry_edge, q.arrival_week)
        for i, q in enumerate(ist.queue_lots)
    ]
    next_id = len(lots)
    pipeline = []
    for s in ist.pipeline:  # initial shipments into a chokepoint get their lot ids at reset, in pipeline order
        into_chk = inst.edges[s.edge].head in inst.chokepoint_ordinal
        pipeline.append(
            Shipment(s.edge, s.k, s.lane, s.qty, s.dispatch_week, s.arrival_week, next_id if into_chk else None)
        )
        next_id += into_chk
    fab_wip: dict[int, dict[int, float]] = {fi: {} for fi in range(len(inst.fabs))}
    for w in ist.fab_wip:
        fi = inst.fab_ordinal[w.node]
        start = w.out_week - inst.nodes[w.node].fab.tau
        fab_wip[fi][start] = fab_wip[fi].get(start, 0.0) + w.qty
    osat_wip: dict[int, dict[int, dict[int, float]]] = {oi: {} for oi in range(len(inst.osats))}
    for w in ist.osat_wip:
        book = osat_wip[inst.osat_ordinal[w.node]].setdefault(w.out_week, {})
        book[w.k] = book.get(w.k, 0.0) + w.qty
    return State(
        week=0,
        stock=initial_stock(inst),
        pipeline=pipeline,
        lots=lots,
        fab_wip=fab_wip,
        osat_wip=osat_wip,
        backlog=np.zeros(len(inst.demands)),
        next_lot_id=next_id,
        last=None,
    )


def lane_queues(lots) -> dict[tuple[int, int, int], float]:
    """Q_ckl of (52): (chokepoint, commodity, lane) -> the lane's lots added in book order, from 0.0.

    ``StepRecord.queue`` records these values, and the chokepoint stock is ``cost.queue_totals`` of them.
    """
    queue: dict[tuple[int, int, int], float] = {}
    for lt in lots:
        key = (lt.chokepoint, lt.k, lt.lane)
        queue[key] = queue.get(key, 0.0) + lt.qty
    return queue


def _new_lot_id(inst: Instance, state: State, e: int) -> int | None:
    """The next lot id for a shipment dispatched on edge e into a chokepoint (§3.4 step 1: at dispatch), else None."""
    if inst.edges[e].head not in inst.chokepoint_ordinal:
        return None
    state.next_lot_id += 1
    return state.next_lot_id - 1


def _take(stock: list[float], s: int, q: float, avail: float, clamps: list, draw: float = 0.0) -> None:
    """stock[s] -= q with the clamp rule of Q58 M11: a result in [-band, 0) becomes 0 and is logged in ``clamps``.

    The band is ``1e-12 max(A-bar, |q|, draw, smallest normal float)`` with A-bar = ``avail``. ``draw`` is what the
    factor that scaled ``q`` was computed from (a dispatch's request capped at the edge's capacity, a grid's requested
    load, an OSAT's throughput): when that factor is subnormal its error is absolute, so ``q`` errs by at most
    ``draw`` 2^-1075, which 1e-12 ``draw`` bounds; below the smallest normal float every operation errs by up to
    2^-1074, so the band never falls below 1e-12 of it (M1 pre-gate ORACLE-PRE-1, INT-1, DET-1). A result below the
    band is a simulator bug.

    Raises:
        RuntimeError: the result is below the band.

    """
    new = stock[s] - q
    if new < 0.0:
        if new < -CLAMP_TOL * max(avail, abs(q), draw, FLOAT_MIN):
            raise RuntimeError(f"stock slot {s} would go to {new!r}: more than the clamp band (Q58 M11)")
        clamps.append((s, new))
        new = 0.0
    stock[s] = new


def step(
    inst: Instance,
    marks: WeeklyMarks,
    state: State,
    flows: dict[int, float],
    overrides: dict[int, float] | None = None,
    holds: frozenset[tuple[int, int]] = frozenset(),
    invalid: tuple[str, ...] = (),
) -> StepRecord:
    """Simulate week ``state.week + 1`` in place and return its record.

    Args:
        inst: the instance.
        marks: the episode's weekly marks.
        state: the state at the end of the previous week; mutated to the end of this week.
        flows: action slot -> requested quantity, already validated (§9.3): finite, >= 0, in range, not masked.
        overrides: override slot -> quantity for tanker cargo K^ov. Any valid override slot or hold for (c, k) turns the
            default release of k at c off this week; a qty-0 slot releases nothing on that slot (Q86).
        holds: (chokepoint node, k) pairs held this week (no release at all).
        invalid: entries the validator dropped, copied into the record's log.

    """
    t = state.week + 1
    if not 1 <= t <= inst.T or marks.T != inst.T:
        raise ValueError(f"week {t} is outside the horizon 1..{inst.T}")
    ti = t - 1
    overrides = {int(s): float(q) for s, q in (overrides or {}).items()}
    flows = {int(s): float(q) for s, q in flows.items()}
    S, F, G, D = len(inst.stock_slots), len(inst.fabs), len(inst.grids), len(inst.demands)
    E = len(inst.edges)
    slot_index = inst.slot_index
    tables: _Tables = inst.memo("dynamics.sim", _Tables.build)
    I = [float(x) for x in state.stock]
    I_prev = list(I)
    u = marks.u[ti].tolist()
    prohibited = marks.prohibited[ti].tolist()
    clamps: list[tuple[int, float]] = []
    edge_clamps: list[tuple[int, float]] = []
    override_clamps: list[tuple[int, float]] = []

    # ----- 3 forward: arrivals join the lot book; overrides first (edge clamp of (4)); default release (10) --------
    chk_mod.arrivals_to_lots(inst, state, t)
    rel = chk_mod.release(inst, state.lots, u, marks.kappa[ti].tolist(), prohibited, overrides, holds, override_clamps)

    # ----- 4 clip (3)-(5), edge clamp on the residual u' (no request edge leaves a chokepoint), fleet slack (7) ----
    released_on: dict[int, float] = {}
    for r in rel:
        released_on[r.edge] = released_on.get(r.edge, 0.0) + r.qty
    cap = [u[e] - released_on[e] if e in released_on else u[e] for e in range(E)]
    x_req = clip_requests(inst, flows, prohibited, cap, I_prev, edge_clamps)
    items = [(*inst.action_slots[s], q) for s, q in x_req.items()]  # (edge, commodity, lane, quantity)
    after = fleet_slack(inst, items + [(r.edge, r.k, r.lane, r.qty) for r in rel], tables.fleet_caps)  # (7)
    for s, q in zip(list(x_req), after[: len(items)]):
        x_req[s] = q
    for r, q in zip(rel, after[len(items) :]):
        r.qty = q

    # ----- 5 dispatch (2): executed requests leave stock, then chokepoint releases leave their lots -----------------
    x_lane: dict[tuple[int, int, int | None], float] = {}
    for s, q in x_req.items():
        if q <= 0:
            continue
        e, k, lane = inst.action_slots[s]
        edge = inst.edges[e]
        tail_slot = slot_index[(edge.tail, k)]
        _take(I, tail_slot, q, I_prev[tail_slot], clamps, min(flows[s], cap[e]))  # the draw: request capped by (4)
        state.pipeline.append(Shipment(e, k, lane, q, t, t + edge.tau, _new_lot_id(inst, state, e)))
        x_lane[(e, k, lane)] = x_lane.get((e, k, lane), 0.0) + q
    ov_exec = {s: 0.0 for s in overrides}
    for r in rel:
        if r.qty <= 0:
            continue
        r.lot.qty -= r.qty
        lid = _new_lot_id(inst, state, r.edge)  # a release onto a lane edge into the next chokepoint (tandem lanes)
        state.pipeline.append(Shipment(r.edge, r.k, r.lane, r.qty, t, t + inst.edges[r.edge].tau, lid))
        x_lane[(r.edge, r.k, r.lane)] = x_lane.get((r.edge, r.k, r.lane), 0.0) + r.qty
        if r.override_slot is not None:
            ov_exec[r.override_slot] += r.qty
    kept = []
    for lt in state.lots:
        if lt.qty > LOT_EPS:
            kept.append(lt)
        elif lt.qty != 0.0:
            clamps.append((slot_index[(lt.chokepoint, lt.k)], lt.qty))
    state.lots = kept
    queue = lane_queues(state.lots)  # Q^t_ckl (52); no lot moves again this week (arrivals at step 6 skip chokepoints)
    chk_stock = queue_totals(queue)  # I^t_ck = fsum over lanes of Q^t_ckl, as the LP forms it (23)
    for s, c, k in tables.chokepoint_slots:
        I[s] = chk_stock.get((c, k), 0.0)

    # ----- 6 arrive at every other node, one shipment at a time in pipeline order ----------------------------------
    keep = []
    for sh in state.pipeline:
        if sh.arrival_week == t:
            I[slot_index[(inst.edges[sh.edge].head, sh.k)]] += sh.qty  # chokepoint arrivals left at step 3 (tau >= 1)
        else:
            keep.append(sh)
    state.pipeline = keep

    # ----- 7 produce ---------------------------------------------------------------------------------------------
    supply = marks.supply[ti].tolist()
    lift = [0.0] * S
    for s, storage in tables.supply_slots:  # refill up to the cap (Q79); the rest of the availability is lost
        q = max(0.0, min(supply[s], storage - I[s]))
        lift[s] = q
        I[s] += q
    alpha, R = marks.alpha_bar[ti].tolist(), marks.R[ti].tolist()
    fab_attrs = tables.fab_attrs
    phat = [
        min(alpha[fi] * R[fi] * fab.cap0, I[slot_index[(f, fab.input)]])
        for fi, (f, fab) in enumerate(zip(inst.fabs, fab_attrs))
    ]
    p = list(phat)
    energy = [0.0] * F
    served_load, shed = [0.0] * G, [0.0] * G
    segment: dict[tuple[int, int | None], float] = {}
    G_bar, y_bar = marks.G_bar[ti].tolist(), marks.y_bar[ti].tolist()
    psi = inst.params.psi
    for gi, g in enumerate(inst.grids):
        grid = inst.nodes[g].grid
        av: dict[int, float] = {}
        for k in grid.fuels:  # (15), (18): segment output after rationing, bounded by fuel on hand after steps 5-6
            s = slot_index[(g, k)]
            if k == grid.rationed:  # min{1, I^{t-1} / (psi I-bar)} as the LP's row psi I-bar G <= zeta G-bar I^{t-1}
                threshold = psi * grid.ibar[k]
                ration = 1.0 if I_prev[s] >= threshold else I_prev[s] / threshold  # psi I-bar = 0 never rations
                av[k] = min(grid.shares[k] * G_bar[gi] * ration, I[s])
            else:
                av[k] = min(grid.shares[k] * G_bar[gi], I[s])
        av_null = grid.shares.get(None, 0.0) * G_bar[gi]
        g_av = 0.0
        for k in grid.fuels:
            g_av += av[k]
        g_av += av_null
        members = inst.grid_fabs[gi]
        e_hat = [(fab_attrs[fi].e * phat[fi] / R[fi] if R[fi] > 0 else 0.0) for fi in members]
        y, E_alloc = allocate_energy(grid.priority, g_av, y_bar[gi], e_hat)
        for fi, ef in zip(members, E_alloc):
            energy[fi] = ef
            if fab_attrs[fi].e > 0:
                p[fi] = min(phat[fi], R[fi] * ef / fab_attrs[fi].e)
        load = (sum(E_alloc) + y) / g_av if g_av > 0 else 0.0
        for k in grid.fuels:
            s = slot_index[(g, k)]
            burn = av[k] * load
            segment[(gi, k)] = burn
            _take(I, s, burn, av[k], clamps, y_bar[gi] + sum(e_hat))  # the draw: the grid's requested load (18)
        segment[(gi, None)] = av_null * load
        served_load[gi] = y
        shed[gi] = y_bar[gi] - y
    scrapped = [0.0] * F
    for fi, (f, fab) in enumerate(zip(inst.fabs, fab_attrs)):
        I[slot_index[(f, fab.input)]] -= p[fi]  # p <= wafers on hand
        wip = state.fab_wip.setdefault(fi, {})
        wip[t] = wip.get(t, 0.0) + p[fi]  # (12): gross WIP booked at start
        for hit in marks.fab_hits:  # (14): scrap of this week's onsets on start weeks [t - w_scr, t - 1]
            if hit.fab == fi and hit.onset_week == t:
                for s0 in range(t - fab.w_scr, t):
                    if s0 in wip:
                        old = wip[s0]
                        wip[s0] = old * (1.0 - hit.severity)
                        scrapped[fi] += old - wip[s0]
        I[slot_index[(f, fab.product)]] += wip.pop(t - fab.tau, 0.0)
    packaged: dict[tuple[int, int], float] = {}
    osat_cap = osat_throughput(inst, marks.R_osat[ti]).tolist()  # (19) under (13): thr_i R_osat_i(t), one home
    for oi, o in enumerate(inst.osats):
        osat = inst.nodes[o].osat
        wip = state.osat_wip.setdefault(oi, {})
        for k, q in sorted(wip.pop(t, {}).items()):  # packaged output joins stock before xi is computed (Q58 M6)
            I[slot_index[(o, k)]] += q
        pairs = sorted(osat.packages.items())
        raw_slots = [slot_index[(o, raw)] for raw, _pk in pairs]
        thr = osat_cap[oi]  # (19) under (13): thr_i R_osat_i(t), the LP's product (§4.4; thr x 1.0 = thr)
        xi = package([I[s] for s in raw_slots], thr)
        book = wip.setdefault(t + osat.tau, {})
        for (_raw, pk), s, q in zip(pairs, raw_slots, xi):
            _take(I, s, q, I[s], clamps, thr)  # A-bar the raw stock; (19) computes xi from thr R_osat
            book[pk] = book.get(pk, 0.0) + q
            packaged[(oi, pk)] = q

    # ----- 8 serve (20) ------------------------------------------------------------------------------------------
    demand = marks.demand[ti].tolist()
    served, lost, backlog = [0.0] * D, [0.0] * D, [0.0] * D
    for di, d in enumerate(inst.demands):
        s = slot_index[(d.node, d.k)]
        want = demand[di] + float(state.backlog[di]) if d.backlog else demand[di]
        served[di] = min(want, I[s])
        I[s] -= served[di]
        if d.backlog:
            backlog[di] = want - served[di]
        else:
            lost[di] = demand[di] - served[di]

    # ----- 9 charge: disposal above I^max at non-chokepoint, non-supply nodes; C_t (23), C^¢_t (24) ------------------
    disposal = [0.0] * S
    for s, storage in tables.disposal_slots:
        if I[s] > storage:
            disposal[s] = I[s] - storage
            I[s] = storage
    costs = weekly_costs(
        inst,
        marks.c[ti].tolist(),
        marks.c_wr[ti].tolist(),
        marks.tariff[ti].tolist(),
        marks.h_queue[ti].tolist(),
        x_lane,  # x_ek = edge_flows(x_lane) (23)
        I,
        disposal,
        lost,
        backlog,
        shed,
    )
    rec = StepRecord(
        week=t,
        requested=dict(sorted(flows.items())),
        executed=dict(x_req),
        override_requested=dict(sorted(overrides.items())),
        override_executed=dict(sorted(ov_exec.items())),
        x=x_lane,
        stock=np.array(I),
        queue=queue,
        disposal=np.array(disposal),
        lift=np.array(lift),
        lots_started=np.array(p),
        scrapped=np.array(scrapped),
        packaged=packaged,
        segment=segment,
        energy=np.array(energy),
        served_load=np.array(served_load),
        shed=np.array(shed),
        demand=np.array(demand),
        served=np.array(served),
        lost=np.array(lost),
        backlog=np.array(backlog),
        costs=costs,
        cost_cents=cents(costs.total()),
        clamps=tuple(clamps),
        invalid=tuple(invalid),
        edge_clamps=tuple(edge_clamps),
        override_clamps=tuple(override_clamps),
    )
    state.stock = rec.stock.copy()
    state.backlog = rec.backlog.copy()
    state.week = t
    state.last = rec
    return rec


def terminal_salvage(inst: Instance, state: State) -> float:
    """Terminal credit S_T of (23) at the end of week T.

    Stock at nu (0 at supply nodes, Q79; a chokepoint's queue at its slot's nu, Q93), shipments in transit at the nu of
    their edge's head, per (edge, commodity, dispatch week) on the week's x_ek (Q93), and fab and OSAT WIP at the
    salvage of their input net of scrap booked by T (Q56).
    """
    if state.week != inst.T:
        raise ValueError(f"the terminal credit is taken at T = {inst.T}, not at week {state.week}")
    return salvage(inst, state)
```

### `shockbench_flow/dynamics/production.py`

```python
"""Step 7 of design §3.2: energy allocation (15)-(18) and OSAT packaging (19) as pure functions (Q27, Q57, Q58 M6).

The simulator (``sim.py``) applies them in the §3.2 order: supply lift, grids (segments, rationing on last week's gas
stock, fuel on hand after steps 5-6, allocation by ``pri_g``, pro-rata segment loading), fab lots with gross WIP, scrap
booked in onset weeks, fab output, OSAT output, then packaging. Divisions by zero follow 0/0 := 0 (§3.5).
"""

from collections.abc import Sequence


def allocate_energy(priority: str, g_av: float, y_bar: float, e_hat: Sequence[float]) -> tuple[float, list[float]]:
    """(18): served base load y and energy E_f per fab of one grid under ``pri_g``.

    Args:
        priority: ``base_first``, ``proportional`` or ``industrial_first`` (Q57).
        g_av: available output G^av_g, the sum over segments after rationing and fuel on hand.
        y_bar: base load y-bar^t_g.
        e_hat: requested draw E-hat_f = e_f p-hat_f / R_f per fab of the grid, in fab order.

    Returns:
        (y, [E_f]); the unserved base load y-bar - y is shed at VOLL (17).

    """
    tot = sum(e_hat)
    if priority == "base_first":
        y = min(y_bar, g_av)
        E = [(eh * min(1.0, (g_av - y) / tot) if tot > 0 else 0.0) for eh in e_hat]
    elif priority == "proportional":
        den = y_bar + tot
        eta = min(1.0, g_av / den) if den > 0 else 0.0
        y = eta * y_bar
        E = [eta * eh for eh in e_hat]
    elif priority == "industrial_first":
        E = [(eh * min(1.0, g_av / tot) if tot > 0 else 0.0) for eh in e_hat]
        y = min(y_bar, max(0.0, g_av - sum(E)))
    else:
        raise ValueError(f"unknown energy priority {priority!r}")
    return y, E


def package(raw: Sequence[float], thr: float) -> list[float]:
    """(19): OSAT starts xi_ik from raw-chip stock, up to the throughput thr_i, pro rata across packaged commodities.

    With one packaged commodity this is ``min(thr, raw)`` exactly, the reference's rule.
    """
    tot = sum(raw)
    if tot <= thr:
        return [float(r) for r in raw]
    return [thr * (r / tot) for r in raw]
```

### `shockbench_flow/dynamics/cost.py`

```python
"""The weekly cost C_t of (23) and the terminal credit S_T (design §3.6; Q3, Q56, Q58, Q68, Q73, Q79, Q93).

Every component is a ``math.fsum`` over explicit terms, with no ``@`` or ``np.dot`` (§3.6); C_t is the fsum of the
components and becomes integer cents once, by (24), in the caller. Flow terms are per (edge, commodity), with x_ek of
(52) the fsum of the week's lane flows x_ekl (``edge_flows``): exact and order-free, so the LP, which holds the same
pieces as lane columns (one aggregated column for a K^ov release), forms the same x_ek and the same cents (53). Tariffs
are charged in the dispatch week at the rate in force then, as ``(rate * v_k) * x_ek`` (Q58 E24, M8).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shockbench_flow.dynamics.state import CostComponents, State
from shockbench_flow.instance.schema import Instance


@dataclass(frozen=True)
class _Tables:
    """Week-invariant cost coefficients of one instance (kept in ``Instance.memo``), in the term order of (23)."""

    v: tuple[float, ...]  # customs value v_k per commodity
    holding: tuple[tuple[int, float], ...]  # (slot, h_ik) at non-chokepoint slots
    queue: tuple[tuple[int, int, int], ...]  # (slot, chokepoint ordinal, k) at chokepoint slots
    disposal: tuple[float, ...]  # c^disp_k per stock slot
    pi: tuple[float, ...]  # pi_ik per demand
    voll: tuple[float, ...]  # VOLL_g per grid ordinal

    @staticmethod
    def build(inst: Instance) -> "_Tables":
        chk = inst.chokepoint_ordinal
        slots = list(enumerate(inst.stock_slots))
        return _Tables(
            v=tuple(com.v for com in inst.commodities),
            holding=tuple((s, sl.holding) for s, sl in slots if sl.node not in chk),
            queue=tuple((s, chk[sl.node], sl.k) for s, sl in slots if sl.node in chk),
            disposal=tuple(inst.commodities[sl.k].disposal_cost for _s, sl in slots),
            pi=tuple(d.pi for d in inst.demands),
            voll=tuple(inst.nodes[g].grid.voll for g in inst.grids),
        )


def edge_flows(x: Mapping[tuple[int, int, int | None], float]) -> dict[tuple[int, int], float]:
    """x_ek of (23), (52): the week's lane flows x_ekl of each (edge, commodity), summed by ``math.fsum``.

    The one formation of x_ek, shared by the simulator (over its lane flows) and the LP (over its lane columns, replay's
    aggregated K^ov column included): fsum is exact and independent of the order of the pieces, so both sides charge
    the same C_t for the same lane flows (design §12, cost terms (23) and oracle cents).

    Args:
        x: (edge, commodity, lane or None) -> flow, in any order.

    Returns:
        (edge, commodity) -> x_ek, in order of first appearance.

    """
    return _fsum_over_lanes(x)


def queue_totals(queue: Mapping[tuple[int, int, int | None], float]) -> dict[tuple[int, int], float]:
    """I_ck at a chokepoint (23), (52): the lane queues Q_ckl of each (chokepoint, commodity), summed by ``math.fsum``.

    The one formation of the chokepoint stock: the simulator applies it to its lane queues (each lane's lots added in
    book order, the values ``StepRecord.queue`` records), the LP to its lane columns and replay to the recorded lane
    queues, so h^Q_ck I_ck is the same float on every side (design §12, cost terms (23)).

    Args:
        queue: (chokepoint, commodity, lane) -> Q_ckl, in any order.

    Returns:
        (chokepoint, commodity) -> I_ck, in order of first appearance.

    """
    return _fsum_over_lanes(queue)


def _fsum_over_lanes(pieces: Mapping[tuple[int, int, int | None], float]) -> dict[tuple[int, int], float]:
    """(a, b, lane) -> q summed over the lanes by ``math.fsum`` per (a, b), exact and independent of the order."""
    groups: dict[tuple[int, int], list[float]] = {}
    for (a, b, _lane), q in pieces.items():
        groups.setdefault((a, b), []).append(float(q))
    return {ab: math.fsum(qs) for ab, qs in groups.items()}


def weekly_costs(
    inst: Instance,
    c: Sequence[float],
    c_wr,
    tariff,
    h_queue,
    x: Mapping[tuple[int, int, int | None], float],
    stock: Sequence[float],
    disposal: Sequence[float],
    lost: Sequence[float],
    backlog: Sequence[float],
    shed: Sequence[float],
) -> CostComponents:
    """The eight components of C_t in (23) for one week.

    Args:
        inst: the instance.
        c: unit freight c^t_e per edge.
        c_wr: per-transit war-risk cost, ``c_wr[e][k]`` (11).
        tariff: ad valorem rate, ``tariff[e][k]``.
        h_queue: queue holding, ``h_queue[chokepoint ordinal][k]``.
        x: (edge, commodity, lane or None) -> executed flow x_ekl of the week, dispatches and chokepoint releases
            (the record's ``x``); the flow terms take x_ek = ``edge_flows(x)``.
        stock: end-of-week I^t per stock slot (queue totals at chokepoints).
        disposal: O^t per stock slot.
        lost: U^t per demand.
        backlog: B^t per demand.
        shed: y^sh,t per grid ordinal.

    """
    tb: _Tables = inst.memo("dynamics.cost", _Tables.build)
    v = tb.v
    x_ek = edge_flows(x)
    return CostComponents(
        freight=math.fsum(c[e] * q for (e, _k), q in x_ek.items()),
        war_risk=math.fsum(c_wr[e][k] * q for (e, k), q in x_ek.items()),
        tariff=math.fsum((tariff[e][k] * v[k]) * q for (e, k), q in x_ek.items()),  # (tau^tar v_k) x_ek (23)
        holding=math.fsum(h * stock[s] for s, h in tb.holding),
        queue_holding=math.fsum(h_queue[ci][k] * stock[s] for s, ci, k in tb.queue),
        shortage=math.fsum(pi * (lost[i] + backlog[i]) for i, pi in enumerate(tb.pi)),
        disposal=math.fsum(dc * disposal[s] for s, dc in enumerate(tb.disposal)),
        shed=math.fsum(voll * shed[i] for i, voll in enumerate(tb.voll)),
    )


def transit_pieces(pipeline) -> dict[int, dict[tuple[int, int, int | None], float]]:
    """P^T of (23) by dispatch week: week -> (edge, commodity, lane) -> the lane's shipments added in pipeline order.

    Each lane piece is added from 0.0 in the order the simulator appended the shipments (dispatches in slot order, then
    chokepoint releases in release order), the formation of the week's x_ekl (``StepRecord.x``); ``edge_flows`` of a
    week's pieces is then the x_ek its flow terms charged, which the LP holds in its week's lane columns (Q93).
    """
    weeks: dict[int, dict[tuple[int, int, int | None], float]] = {}
    for sh in pipeline:
        book = weeks.setdefault(sh.dispatch_week, {})
        key = (sh.edge, sh.k, sh.lane)
        book[key] = book.get(key, 0.0) + sh.qty
    return weeks


def salvage(inst: Instance, state: State) -> float:
    """S_T of (23) from the end-of-horizon state (Q3, Q56, Q79, Q93).

    Stock at nu_ik, a chokepoint's queue total (the fsum of its lane queues, ``queue_totals``) at the chokepoint slot's
    nu = c^min x the commodity's share (Q93), 0 at supply nodes (Q79); shipments still in transit at the nu of their
    edge's head, where they are going (Q93), one term per (edge, commodity, dispatch week): nu_head times x_ek, the
    fsum over lanes of the week's lane pieces (``transit_pieces``, ``edge_flows``), the x_ek its flow terms charged, so
    the LP forms the same float from its lane columns; fab WIP at the salvage of its wafer input and OSAT WIP at the
    salvage of its raw-chip input, both net of the scrap booked by T (the ledger is net once booked, Q56).
    """
    supply = set(inst.supply_nodes)

    def nu(node: int, k: int) -> float:
        if node in supply:
            return 0.0
        s = inst.slot_index.get((node, k))
        return 0.0 if s is None else inst.stock_slots[s].salvage

    terms = [nu(sl.node, sl.k) * float(state.stock[s]) for s, sl in enumerate(inst.stock_slots)]
    for pieces in transit_pieces(state.pipeline).values():
        terms += [nu(inst.edges[e].head, k) * x for (e, k), x in edge_flows(pieces).items()]
    for fi, f in enumerate(inst.fabs):
        rate = nu(f, inst.nodes[f].fab.input)
        terms += [rate * q for q in state.fab_wip.get(fi, {}).values()]
    for oi, o in enumerate(inst.osats):
        raw_of = {pk: raw for raw, pk in inst.nodes[o].osat.packages.items()}
        for book in state.osat_wip.get(oi, {}).values():
            terms += [nu(o, raw_of[k]) * q for k, q in book.items()]
    return math.fsum(terms)
```

### `shockbench_flow/dynamics/chokepoint.py`

```python
"""The chokepoint lot book and its weekly release, design §3.4 pseudocode steps 1-3, (9)-(10) (Q4, Q25, Q34, Q68, Q86).

Step 1: shipments arriving at a chokepoint in week t become lots under the lot id they received at dispatch (in
action-slot order, then release order for tandem-lane releases; initial shipments at reset in pipeline order).

Step 2, overrides first (tanker cargo K^ov only): any override slot or hold for (c, k) turns the default release of k at
c off this week. A hold releases nothing; when a hold and override slots name the same (c, k), the hold wins. Override
quantities pass (3) (K_e and Z_t), (4) the out-edge capacity per edge over its override slots, (5) the queue content of
k at c and (6) the pool's throughput kappa_cb, each factor pro rata across the override slots it covers. On an out-edge
whose float factor (4) is subnormal, the override slots then pass the edge clamp of the dispatch clip in slot order,
before (6) (``clip.edge_clamp``, Q95): a residual of u'_e left in [-b, 0), with b = 1e-12 max(u'_e, |x|, q, smallest
normal float) and q the slot's override request, lowers that slot to the residual before it and is logged as
(override slot, value) (``StepRecord.override_clamps``); anything lower raises ``RuntimeError``. Units are then taken
FIFO within k by (arrival_week, dispatch_week, entry_edge, lane, lot_id); the taken part ships with the slot's lane
(None on a turn-back on no lane) and the remainder of a split lot keeps its id and its place in the book.

Step 3, the default release (10) on the residual kappa'_cb and u'_e: arrival cohorts in increasing order and, inside a
cohort, the pools (tb, ct); a lot whose next edge is prohibited for its commodity releases nothing; next-edge capacity
pro rata inside the cohort, then throughput pro rata inside the cohort. The arithmetic follows the design's reference
code (``tiny_fixture.py``, ``Sim.step`` step 3) exactly, including its residual updates ``u' -= X`` per released lot and
``kappa' -= eta_kappa * total``, so the package reproduces its cents.

Releases returned here are tentative: the simulator applies the fleet slack (7) last, jointly with the week's duplicate
dispatches, and only then takes the released quantities out of the lots; the unreleased share stays in the lot it came
from, which keeps its key (§3.4 step 4).
"""

from collections import defaultdict
from dataclasses import dataclass

from shockbench_flow.dynamics.clip import edge_stock_clip
from shockbench_flow.dynamics.state import Lot, State
from shockbench_flow.instance.schema import Instance


@dataclass
class Release:
    """A tentative release of part of one lot onto an out-edge of its chokepoint."""

    lot: Lot
    edge: int
    k: int
    lane: int | None
    qty: float
    override_slot: int | None  # None for the default release


def fifo_key(lot: Lot) -> tuple[int, int, int, int, int]:
    """The FIFO order of overrides (§3.4 step 2)."""
    return (lot.arrival_week, lot.dispatch_week, lot.entry_edge, lot.lane, lot.lot_id)


def arrivals_to_lots(inst: Instance, state: State, t: int) -> None:
    """Step 1: shipments arriving at a chokepoint in week t leave the pipeline and join its lot book."""
    chk = inst.chokepoint_ordinal
    keep = []
    for s in state.pipeline:
        head = inst.edges[s.edge].head
        if s.arrival_week == t and head in chk:
            nxt = None if s.lane is None else inst.lane_next_edge(s.lane, s.edge)
            if nxt is None:
                raise ValueError(
                    f"shipment on edge {inst.edges[s.edge].id} reaches a chokepoint without a lane to follow"
                )
            if s.lot_id is None:
                raise ValueError(f"shipment on edge {inst.edges[s.edge].id} reaches a chokepoint without a lot id")
            state.lots.append(Lot(s.lot_id, head, s.k, s.qty, s.lane, nxt, s.dispatch_week, s.edge, t))
        else:
            keep.append(s)
    state.pipeline = keep


def release(
    inst: Instance,
    lots: list[Lot],
    u: list[float],
    kappa: list[list[float]],
    prohibited,
    overrides: dict[int, float],
    holds: frozenset[tuple[int, int]],
    clamps: list[tuple[int, float]] | None = None,
) -> list[Release]:
    """Steps 2-3 at every chokepoint (node order): overrides first, then the default release (10).

    Args:
        inst: the instance.
        lots: the lot book after this week's arrivals (not modified here).
        u: u^t_e per edge.
        kappa: kappa^t_cb per chokepoint ordinal and pool.
        prohibited: Z_t, indexable as ``prohibited[e][k]``.
        overrides: override slot -> quantity (validated); every slot turns the default release of its (c, k) off.
        holds: (chokepoint node, k) held this week.
        clamps: if given, each edge clamp of an override slot (Q95; ``_overrides``) is appended to it as (override
            slot, value), value < 0, per chokepoint in slot order. The clamp acts whether or not it is logged.

    Returns:
        Tentative releases in release order: per chokepoint, override pieces in slot order, then default releases by
        cohort, pool and book order.

    Raises:
        RuntimeError: an out-edge residual below the clamp band (``clip.edge_clamp``).

    """
    pool = inst.commodity_pool
    off = set(holds) | {inst.override_slots[s][:2] for s in overrides}
    books: dict[int, list[Lot]] = defaultdict(list)  # chokepoint node -> its lots in book order
    for lt in lots:
        books[lt.chokepoint].append(lt)
    override_slots: dict[int, list[int]] = defaultdict(list)  # chokepoint node -> its override slots in slot order
    for s in sorted(overrides):
        override_slots[inst.override_slots[s][0]].append(s)
    ures = list(u)
    out: list[Release] = []
    for ci, c in enumerate(inst.chokepoints):
        kap = list(kappa[ci])
        book = books.get(c, [])
        # ----- step 2: overrides ---------------------------------------------------------------------------------
        if slots := override_slots.get(c):
            _overrides(inst, c, book, slots, overrides, holds, prohibited, ures, kap, out, clamps)
        # ----- step 3: default release (10), FIFO across cohorts, pro rata inside a cohort ------------------------
        cohorts: dict[tuple[int, int], list[Lot]] = defaultdict(list)  # (arrival week, pool) -> lots in book order
        for lt in book:
            if lt.qty > 0 and (c, lt.k) not in off:
                cohorts[(lt.arrival_week, pool[lt.k])].append(lt)
        for a, b in sorted(cohorts):
            coh = cohorts[(a, b)]
            # Y-tilde of (10): a lot releases nothing onto a next edge that is prohibited for it or does not permit it
            y = [
                0.0 if prohibited[lt.next_edge][lt.k] or lt.k not in inst.edges[lt.next_edge].K else lt.qty
                for lt in coh
            ]
            tot: dict[int, float] = {}
            for lt, yv in zip(coh, y):
                tot[lt.next_edge] = tot.get(lt.next_edge, 0.0) + yv
            eta_u = {e: (min(1.0, ures[e] / s) if s > 0 else 0.0) for e, s in tot.items()}
            tot2 = 0.0
            for lt, yv in zip(coh, y):
                tot2 += eta_u[lt.next_edge] * yv
            eta_k = min(1.0, kap[b] / tot2) if tot2 > 0 else 0.0
            for lt, yv in zip(coh, y):
                q = eta_k * eta_u[lt.next_edge] * yv
                if q > 0:
                    out.append(Release(lt, lt.next_edge, lt.k, lt.lane, q, None))
                    ures[lt.next_edge] -= q
            kap[b] -= eta_k * tot2
    return out


def _overrides(
    inst: Instance,
    c: int,
    book: list[Lot],
    slots: list[int],
    overrides: dict[int, float],
    holds: frozenset[tuple[int, int]],
    prohibited,
    ures: list[float],
    kap: list[float],
    out: list[Release],
    clamps: list[tuple[int, float]] | None = None,
) -> None:
    """Step 2 at chokepoint c: override pieces in slot order, FIFO within k; updates ``ures``, ``kap`` and ``out``.

    (3)-(5) are the clip of the policy's requests (``clip.edge_stock_clip``) with the queue content of k at c, this
    week's arrivals included, as the stock, and its edge clamp on the out-edges whose float factor (4) is subnormal
    (Q95), logged in ``clamps`` (if given) as (override slot, value); (6) then scales each pool to the residual
    throughput. (6) and the fleet slack (7) only lower a slot's quantity, so the clamp's bound holds after them.
    """
    pool = inst.commodity_pool
    live = []  # (slot, k, edge, lane) of the entries that pass the mask (3)
    items = []
    for s in slots:
        _c, k, e, lane = inst.override_slots[s]
        q = overrides[s]
        if (c, k) not in holds and q > 0 and k in inst.edges[e].K and not prohibited[e][k]:  # (3)
            live.append((s, k, e, lane))
            items.append((e, k, (c, k), q))
    content: dict[tuple[int, int], float] = {}
    for key in {key for _e, _k, key, _q in items}:
        content[key] = sum(lt.qty for lt in book if lt.k == key[1])
    log: list[tuple[int, float]] = []
    clipped = edge_stock_clip(items, ures, content, log)  # (4) with the edge clamp (Q95), (5)
    if clamps is not None:
        clamps.extend((live[i][0], value) for i, value in log)
    by_b = [0.0, 0.0]
    for (_s, k, _e, _lane), x in zip(live, clipped):
        by_b[pool[k]] += x
    f6 = [min(1.0, max(0.0, kap[b]) / by_b[b]) if by_b[b] > 0 else 0.0 for b in (0, 1)]  # (6)
    req = [(s, k, e, lane, x * f6[pool[k]]) for (s, k, e, lane), x in zip(live, clipped)]
    left = {lt.lot_id: lt.qty for lt in book}
    fifo = sorted(book, key=fifo_key)
    for s, k, e, lane, take in req:
        if take <= 0:
            continue
        ures[e] -= take
        kap[pool[k]] -= take
        for lt in fifo:
            if take <= 0:
                break
            if lt.k != k or left[lt.lot_id] <= 0:
                continue
            piece = min(take, left[lt.lot_id])
            left[lt.lot_id] -= piece
            take -= piece
            out.append(Release(lt, e, k, lane, piece, s))
```

### `shockbench_flow/dynamics/clip.py`

```python
"""The clip of the policy's requests, (3)-(5), and the fleet slack (7) (design §3.3; Q2, Q42, Q49, Q68 E3).

Requests are indexed by action slot (edge, commodity, lane). (3) masks slots whose commodity the edge does not permit or
whose pair is in Z_t; (4) caps each edge's total over its requests at the residual capacity u'_e; (5) scales every
(edge, commodity) total drawing on one stock (tail node, commodity) to that stock, and applies the per-(edge, commodity)
factors pro rata to every lane sub-request. Requests are processed in slot order, so the result is independent of the
order in which the policy listed them (F1 evidence ``f1_clip_order.py``). Each factor is at most 1.

Floating-point order follows the design's reference code (``scripts/python/evidence/tiny_fixture.py``, ``Sim.step``
step 4): sequential sums starting from 0.0, the edge factor ``min(1, u / total)``, the stock factor
``min(1, A / total)`` and the executed quantity ``(q * f_edge) * f_stock``; the fleet factor is applied as
``q * cap / total`` when the pool binds (7). When a sum of requests overflows the float range (requests near the float
maximum, which no reference run makes), (4)-(5) are computed in exact rational arithmetic and each executed quantity is
rounded once (``_exact_clip``); so is (7) for a pool whose total overflows or whose binding product ``q * cap`` leaves
the normal float range (``fleet_slack``; M1 pre-gate 3, DOC-1).

Edge clamp (Q95): with a request near the float maximum on an edge of small capacity, the edge factor of (4) is
subnormal and errs by up to 2^-1075 absolutely, so ``q * f_edge`` can put the edge's executed total above u'_e by far
more than 1e-12 relative (docs/004 §3.1: 1.1955e307 on E11 at 1200 x 2^-52 executed 2.664535259103041e-13 against
2.6645352591003757e-13). On an edge whose factor (4) is subnormal, ``clip_requests`` therefore takes the executed
quantities from u'_e in slot order under the stock clamp rule of ``sim._take`` (Q58 M11, design §12 'Stock clamps and
dust lots'): a residual left in [-b, 0), with b = 1e-12 max(u'_e, |x|, q, smallest normal float) and q the slot's
request (the draw from which its share of (4) was computed), sets that slot to the residual before it and is logged as
(slot, value); anything lower raises ``RuntimeError`` (``edge_clamp``). A normal factor errs by a few units in the last
place, as it always did (inside the tolerance of the replay (53)), and is left alone, so no run with a normal factor
moves; nor does the exact path, which rounds each quantity once. The override step at chokepoints
(``chokepoint._overrides``) computes its (4)-(5) with ``edge_stock_clip`` too and passes a log, so its override slots
pass the same clamp on an out-edge whose factor (4) is subnormal (a probe released 1.28e-3 relative above u'_e there
before), logged by override slot in ``StepRecord.override_clamps``.

The fleet slack (7) sums ``Delta tau * x`` over the terms of ``Instance.dup_items`` (edge, lane or None, Delta tau): a
term with lane None, an edge-level sea duplicate or the turn-back where a duplicate lane leaves its route at a
chokepoint, matches every flow on its edge; a term with a lane, a sea duplicate lane leaving its route at its entry
edge, matches only the flow that edge carries on that lane (design §3.3: "Cape, Lombok-Sunda and east-of-Taiwan lanes
and turn-back edges into them"). Every matching release, dispatch or chokepoint release, is scaled when its pool binds.

Residual edge capacity (general rule): u'_e = u^t_e minus the chokepoint releases on e this week. Action slots never
leave a chokepoint (§12, action slots), so on every instance u'_e = u^t_e for every edge that carries a request; the
reference computes its chokepoint residuals separately for the same reason.
"""

import math
import sys
from collections.abc import Container, Iterable, Mapping, Sequence
from fractions import Fraction

from shockbench_flow.instance.schema import Instance


CLAMP_TOL = 1e-12  # relative clamp band [-1e-12 max(A-bar, draw), 0) of Q58 M11, for stocks (sim) and edges (Q95)
FLOAT_MIN = sys.float_info.min  # the band's floor: below the smallest normal float, errors are absolute (2^-1074)


def edge_stock_clip(
    items: Sequence[tuple[int, int, tuple[int, int], float]],
    cap: Sequence[float],
    avail: Mapping[tuple[int, int], float],
    clamps: list[tuple[int, float]] | None = None,
) -> list[float]:
    """(4)-(5) for requests that already passed the mask (3): joint edge cap, then shared stock pro rata.

    Shared by the clip of the policy's requests and the override step at chokepoints (§3.4 step 2), so both apply the
    same arithmetic.

    Args:
        items: (edge, commodity, stock key, quantity) in processing order; the stock key identifies the stock A-bar the
            request draws on, (tail node, commodity).
        cap: residual capacity u'_e per edge.
        avail: A-bar per stock key.
        clamps: if given, the edges whose float factor (4) is subnormal pass the edge clamp (``edge_clamp``, Q95),
            whose entries (item index, value) are appended to it; if None, nothing is clamped.

    Returns:
        The executed quantity of each item, ``(q * f_edge) * f_stock``, after the edge clamp when ``clamps`` is given.

    Raises:
        RuntimeError: an edge residual below the clamp band (``edge_clamp``).

    """
    by_e: dict[int, float] = {}
    by_ek: dict[tuple[int, int], float] = {}
    for e, k, _key, q in items:
        by_e[e] = by_e.get(e, 0.0) + q
        by_ek[(e, k)] = by_ek.get((e, k), 0.0) + q
    if not all(math.isfinite(v) for v in by_e.values()):
        return _exact_clip(items, cap, avail)
    f_edge = {e: (min(1.0, max(0.0, cap[e]) / tot) if tot > 0 else 0.0) for e, tot in by_e.items()}  # (4)
    key_of = {(e, k): key for e, k, key, _q in items}
    by_key: dict[tuple[int, int], float] = {}
    for (e, k), tot in by_ek.items():
        key = key_of[(e, k)]
        by_key[key] = by_key.get(key, 0.0) + tot * f_edge[e]
    if not all(math.isfinite(v) for v in by_key.values()):
        return _exact_clip(items, cap, avail)
    f_stock = {key: (min(1.0, avail.get(key, 0.0) / tot) if tot > 0 else 0.0) for key, tot in by_key.items()}  # (5)
    x = [q * f_edge[e] * f_stock[key] for e, _k, key, q in items]
    if clamps is not None and (sub := {e for e, f in f_edge.items() if 0.0 < f < FLOAT_MIN}):
        x = edge_clamp(items, x, cap, sub, clamps)
    return x


def _exact_clip(
    items: Sequence[tuple[int, int, tuple[int, int], float]],
    cap: Sequence[float],
    avail: Mapping[tuple[int, int], float],
) -> list[float]:
    """(4)-(5) in exact rational arithmetic, for requests near the float maximum whose sums overflow.

    Every factor is the exact ``min(1, a / total)`` and every executed quantity ``q f_edge f_stock`` is rounded once
    to the nearest float, so the executed total on an edge or a stock exceeds its bound by at most half a unit in the
    last place per item. An infinite capacity never binds, nor does an infinite stock, as on the float path: a queue
    content of the override step whose float sum overflowed (lots near the float maximum; it raised ``OverflowError``
    here before, Q95). Scaling all quantities by a power of two instead rounded small (subnormal) stocks and requests,
    by up to 2^17 units in the last place (M1 pre-gate, INT-1 follow-up).
    """
    by_e: dict[int, Fraction] = {}
    by_ek: dict[tuple[int, int], Fraction] = {}
    for e, k, _key, q in items:
        by_e[e] = by_e.get(e, 0) + Fraction(q)
        by_ek[(e, k)] = by_ek.get((e, k), 0) + Fraction(q)
    f_edge: dict[int, Fraction] = {}
    for e, tot in by_e.items():  # (4)
        if tot <= 0:
            f_edge[e] = Fraction(0)
        elif cap[e] == math.inf:
            f_edge[e] = Fraction(1)
        else:
            f_edge[e] = min(Fraction(1), Fraction(max(0.0, cap[e])) / tot)
    key_of = {(e, k): key for e, k, key, _q in items}
    by_key: dict[tuple[int, int], Fraction] = {}
    for (e, k), tot in by_ek.items():
        key = key_of[(e, k)]
        by_key[key] = by_key.get(key, 0) + tot * f_edge[e]
    f_stock: dict[tuple[int, int], Fraction] = {}
    for key, tot in by_key.items():  # (5)
        a = avail.get(key, 0.0)
        if tot <= 0:
            f_stock[key] = Fraction(0)
        elif a == math.inf:
            f_stock[key] = Fraction(1)
        else:
            f_stock[key] = min(Fraction(1), Fraction(a) / tot)
    return [float(Fraction(q) * f_edge[e] * f_stock[key]) for e, _k, key, q in items]


def edge_clamp(
    items: Sequence[tuple[int, int, tuple[int, int], float]],
    executed: Sequence[float],
    cap: Sequence[float],
    edges: Container[int],
    clamps: list[tuple[int, float]],
) -> list[float]:
    """The edge clamp of (4) (Q95): on each edge in ``edges``, the executed quantities never overdraw u'_e.

    The rule of ``sim._take`` with u'_e in place of a stock: the residual starts at ``max(0, cap[e])`` and each item on
    the edge takes its executed quantity x from it in the given order; a result in [-b, 0), with
    b = ``1e-12 max(u'_e, |x|, q, smallest normal float)`` and q the item's request (the draw its share of (4) was
    computed from; a subnormal edge factor errs by up to ``q 2^-1075``), sets x to the residual before the take (so the
    residual becomes exactly 0) and is logged in ``clamps`` as (item index, value), value < 0. An infinite capacity
    never binds. Items on other edges, and every item where no residual goes negative, keep ``executed`` bit for bit.

    Args:
        items: (edge, commodity, stock key, request q) in processing order, as ``edge_stock_clip`` takes them.
        executed: the executed quantity of each item after (4)-(5).
        cap: residual capacity u'_e per edge.
        edges: the edges to clamp (``edge_stock_clip``: those whose float factor (4) is subnormal).
        clamps: list the clamps are appended to.

    Returns:
        The executed quantity of each item after the clamp.

    Raises:
        RuntimeError: a residual below the band (a clip bug; no valid request reaches it).

    """
    out = list(executed)
    resid: dict[int, float] = {}
    for i, ((e, _k, _key, q), x) in enumerate(zip(items, executed)):
        if e not in edges:
            continue
        bound = max(0.0, cap[e])
        r = resid.get(e, bound)
        new = r - x
        if new < 0.0:
            if new < -CLAMP_TOL * max(bound, abs(x), q, FLOAT_MIN):
                raise RuntimeError(f"edge {e} would be left at {new!r} of its capacity: more than the clamp band (Q95)")
            clamps.append((i, new))
            out[i], new = r, 0.0
        resid[e] = new
    return out


def clip_requests(
    inst: Instance,
    requests: Mapping[int, float],
    prohibited,
    cap: Sequence[float],
    avail: Sequence[float],
    clamps: list[tuple[int, float]] | None = None,
) -> dict[int, float]:
    """(3)-(5): mask, joint edge cap, shared stock pro rata over the lane sub-requests of each (edge, commodity).

    Edges whose factor (4) is subnormal then pass the edge clamp in slot order (``edge_clamp``, Q95).

    Args:
        inst: the instance.
        requests: action slot -> requested quantity (already validated, §9.3).
        prohibited: Z_t of the week, indexable as ``prohibited[e][k]`` (bool).
        cap: residual capacity u'_e per edge.
        avail: A-bar of (5) per stock slot: I^{t-1} at the tail nodes (never chokepoints, which have no action slot).
        clamps: if given, each edge clamp is appended to it as (action slot, value), value < 0, in slot order.

    Returns:
        Action slot -> executed quantity for every requested slot, in slot order (0 for masked or non-positive ones).

    Raises:
        RuntimeError: an edge residual below the clamp band (``edge_clamp``).

    """
    out: dict[int, float] = {}
    live: list[int] = []
    items: list[tuple[int, int, tuple[int, int], float]] = []
    for s in sorted(requests):
        q = requests[s]
        e, k, _lane = inst.action_slots[s]
        out[s] = 0.0
        if q > 0 and k in inst.edges[e].K and not prohibited[e][k]:  # (3)
            live.append(s)
            items.append((e, k, (inst.edges[e].tail, k), q))
    stock = {key: avail[inst.slot_index[key]] for _e, _k, key, _q in items if key in inst.slot_index}
    log: list[tuple[int, float]] = []
    for s, x in zip(live, edge_stock_clip(items, cap, stock, log)):
        out[s] = x
    if clamps is not None:
        clamps.extend((live[i], value) for i, value in log)
    return out


def fleet_caps(inst: Instance) -> tuple[float, float]:
    """s^b_fl F^b per pool (tb, ct), the right-hand side of the LP row of (7)."""
    return tuple(share * measure for share, measure in zip(inst.params.fleet_share, inst.params.fleet_measure))


def dup_terms(inst: Instance) -> Mapping[int, tuple[tuple[int | None, int], ...]]:
    """The fleet-slack terms of (7) by edge, from ``inst.dup_items``: edge -> ((lane or None, Delta tau), ...).

    A term with lane None (an edge-level sea duplicate or turn-back) matches every flow on its edge; a term with a lane
    (a sea duplicate lane leaving its route at its entry edge) matches only that lane's flow on it (design §3.3).
    """
    return inst.memo("dynamics.clip.dup_terms", _build_dup_terms)


def _build_dup_terms(inst: Instance) -> dict[int, tuple[tuple[int | None, int], ...]]:
    terms: dict[int, list[tuple[int | None, int]]] = {}
    for e, lane, dtau in inst.dup_items:
        terms.setdefault(e, []).append((lane, dtau))
    return {e: tuple(v) for e, v in terms.items()}


def on_dup(inst: Instance, e: int, lane: int | None) -> bool:
    """Whether a release on edge ``e`` carrying ``lane`` matches a fleet-slack term of (7), so the pool scales it."""
    return any(ln is None or ln == lane for ln, _dtau in dup_terms(inst).get(e, ()))


def fleet_totals(inst: Instance, items: Iterable[tuple[int, int, int | None, float]]) -> list[float]:
    """Sum of Delta tau x over the fleet-slack terms of (7) per pool, accumulated in the given order.

    Args:
        inst: the instance.
        items: (edge, commodity, lane or None, quantity) of every release of the week: dispatches first (slot order),
            then chokepoint releases (release order). An item adds ``Delta tau * q`` for every term of
            ``inst.dup_items`` it matches (``dup_terms``); items that match none are ignored.

    """
    terms = dup_terms(inst)
    tot = [0.0, 0.0]
    for e, k, lane, q in items:
        for ln, dtau in terms.get(e, ()):
            if ln is None or ln == lane:
                tot[inst.commodities[k].pool_index] += dtau * q
    return tot


def fleet_scaled(q: float, total: float, cap: float) -> float:
    """One duplicate release after (7): unchanged unless its pool binds, else ``q * cap / total`` (reference order)."""
    return q * cap / total if total > cap else q


def fleet_slack(
    inst: Instance, items: Sequence[tuple[int, int, int | None, float]], caps: Sequence[float]
) -> list[float]:
    """(7): every release of the week after the fleet slack, in the order given.

    Each pool sums ``Delta tau * q`` over the terms its releases match (``fleet_totals``, in the given order); when the
    total exceeds ``caps[b]`` = s^b_fl F^b, every matching release of the pool becomes ``q * cap / total`` in the
    reference order (``fleet_scaled``). A pool whose float total is not finite, or whose binding product ``q * cap``
    leaves the normal float range (overflows, or underflows from nonzero factors), is computed in exact rational
    arithmetic instead: the exact total, binding when it exceeds a finite cap, and each scaled release
    ``q * cap / total`` rounded once to the nearest float, so no release is ever scaled up. Floats cannot do either
    there: a total of inf executed 0, a product of inf an infinite shipment, and a subnormal product rounds up to twice
    ``q`` (M1 pre-gate 3, DOC-1, ORACLE-PRE3-2).

    Args:
        inst: the instance.
        items: (edge, commodity, lane or None, quantity) of every tentative release of the week: dispatches first (slot
            order), then chokepoint releases (release order), as ``fleet_totals`` takes them.
        caps: s^b_fl F^b per pool (``fleet_caps``).

    Returns:
        The quantity of each item after (7); items that match no term of ``inst.dup_items`` are unchanged.

    """
    terms = dup_terms(inst)
    totals = fleet_totals(inst, items)
    out = [q for _e, _k, _lane, q in items]
    for b, (total, cap) in enumerate(zip(totals, caps)):
        pool = inst.commodity_pool
        members = [i for i, (e, k, lane, _q) in enumerate(items) if pool[k] == b and on_dup(inst, e, lane)]
        if math.isfinite(total) and (total <= cap or all(_normal_product(out[i], cap) for i in members)):
            for i in members:  # the reference arithmetic
                out[i] = fleet_scaled(out[i], total, cap)
            continue
        exact = Fraction(0)  # the same terms as fleet_totals, summed exactly
        for i in members:
            e, _k, lane, q = items[i]
            exact += sum(dtau for ln, dtau in terms[e] if ln is None or ln == lane) * Fraction(q)
        if not math.isfinite(cap) or exact <= Fraction(cap):
            continue  # an infinite cap never binds, as in (4)
        for i in members:
            out[i] = float(Fraction(out[i]) * Fraction(cap) / exact)
    return out


def _normal_product(q: float, cap: float) -> bool:
    """Whether ``q * cap`` keeps the float relative accuracy: finite, and normal unless a factor is 0."""
    p = q * cap
    return math.isfinite(p) and (p >= sys.float_info.min or q == 0.0 or cap == 0.0)
```

### `shockbench_flow/dynamics/state.py`

```python
"""Simulator state, the per-week record, the trajectory and integer cents (design §3.2-3.6, §9.1; Q68, V1, B2).

The ``StepRecord`` holds every realised quantity of one week, so the replay test (53) can feed a trajectory into the
rows of the LP (51) and the cost can be recomputed. Integer codes: stock slots S, action slots, override slots, and the
ordinals of chokepoints, grids G, fabs F, OSATs and demands D of the instance.
"""

import math
from dataclasses import dataclass, field
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)

import numpy as np

from shockbench_flow.instance.io import sha256_hex


# (24) runs under this private context, never the ambient one, so no decimal setting of the caller (precision, rounding,
# traps, or decimal.DefaultContext, from which Context() copies any field left out) changes a cent (DET-1). A float
# repr has at most 17 significant digits and the largest finite float 309 integer digits (311 after the * 100), so 400
# digits hold every product and every quantized result exactly.
_CENTS_CONTEXT = Context(
    prec=400,
    rounding=ROUND_HALF_EVEN,
    Emin=MIN_EMIN,
    Emax=MAX_EMAX,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
_ONE = Decimal(1)


def cents(x: float) -> int:
    """(24): round_half_even(100 * Decimal(repr(x))) — the one conversion from USD floats to integer cents.

    Every step uses the methods of the module's own context, so the result is independent of
    ``decimal.getcontext()``. A non-finite ``x`` has no cents: ``decimal.InvalidOperation`` (inf) or ``ValueError``
    (nan).
    """
    ctx = _CENTS_CONTEXT
    return int(ctx.quantize(ctx.multiply(ctx.create_decimal(repr(float(x))), 100), _ONE))


COST_COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


@dataclass(frozen=True)
class CostComponents:
    """The terms of C_t in (23), USD, each a ``math.fsum`` over explicit terms (no ``@`` or ``np.dot``; §3.6)."""

    freight: float = 0.0
    war_risk: float = 0.0
    tariff: float = 0.0
    holding: float = 0.0
    queue_holding: float = 0.0
    shortage: float = 0.0
    disposal: float = 0.0
    shed: float = 0.0

    def total(self) -> float:
        """C_t = fsum of the components."""
        return math.fsum(getattr(self, k) for k in COST_COMPONENTS)

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in COST_COMPONENTS}


@dataclass
class Shipment:
    """A shipment in the pipeline: dispatched on ``edge`` in ``dispatch_week``, arriving in ``arrival_week`` (2)."""

    edge: int
    k: int
    lane: int | None  # the lane declared at dispatch (kept after a chokepoint release); None off lanes
    qty: float
    dispatch_week: int
    arrival_week: int
    lot_id: int | None = None  # set at dispatch for shipments into a chokepoint (§3.4 step 1), in action-slot order


@dataclass
class Lot:
    """A queue lot at a chokepoint (§3.4 pseudocode, step 1)."""

    lot_id: int
    chokepoint: int  # node index
    k: int
    qty: float
    lane: int
    next_edge: int
    dispatch_week: int
    entry_edge: int
    arrival_week: int


@dataclass
class State:
    """Mutable simulator state at the end of week ``week`` (0 at reset); the next step simulates week ``week + 1``."""

    week: int
    stock: np.ndarray  # (S,) I^t per stock slot; at a chokepoint slot the queue total, the sum of its lots
    pipeline: list[Shipment]
    lots: list[Lot]
    fab_wip: dict[int, dict[int, float]]  # fab ordinal -> start week -> gross remaining lots (mature at start + tau)
    osat_wip: dict[int, dict[int, dict[int, float]]]  # osat ordinal -> out week -> packaged k -> qty
    backlog: np.ndarray  # (D,) B^t per demand
    next_lot_id: int = 0
    last: "StepRecord | None" = None  # the record of week ``week`` (feeds ``last_week`` of the observation)


_WHEN_LOGGED = ("edge_clamps", "override_clamps")  # StepRecord fields in as_json only when not empty (Q95)


@dataclass(frozen=True)
class StepRecord:
    """Everything realised in one week (§3.2 steps 3-9), in the instance's integer codes."""

    week: int
    requested: dict[int, float]  # action slot -> requested qty after the validity rules (§9.3)
    executed: dict[int, float]  # action slot -> executed dispatch after the clip (3)-(7)
    override_requested: dict[int, float]  # override slot -> requested
    override_executed: dict[int, float]  # override slot -> released
    x: dict[tuple[int, int, int | None], float]  # (edge, k, lane) -> x^t_ekl: dispatches and chokepoint releases
    stock: np.ndarray  # (S,) end-of-week I^t (chokepoint slots: queue totals)
    queue: dict[tuple[int, int, int], float]  # (chokepoint node, k, lane) -> end-of-week queue content, (52)
    disposal: np.ndarray  # (S,) O^t
    lift: np.ndarray  # (S,) varsigma^t, 0 at non-supply slots (Q79)
    lots_started: np.ndarray  # (F,) p^t_f
    scrapped: np.ndarray  # (F,) gross WIP scrapped in this onset week (14), booked by the simulator (Q44, Q69)
    packaged: dict[tuple[int, int], float]  # (osat ordinal, packaged k) -> xi^t_ik (19)
    segment: dict[tuple[int, int | None], float]  # (grid ordinal, fuel k or None for ∅) -> G^t_gk (15), (18)
    energy: np.ndarray  # (F,) E^t_f; 0 at fabs without a grid
    served_load: np.ndarray  # (G,) y^t_g
    shed: np.ndarray  # (G,) y^sh,t_g
    demand: np.ndarray  # (D,) d^t
    served: np.ndarray  # (D,) D^t
    lost: np.ndarray  # (D,) U^t (0 at backlog sinks)
    backlog: np.ndarray  # (D,) B^t (0 at lost-sales sinks)
    costs: CostComponents
    cost_cents: int  # C^¢_t (24)
    # (stock slot, value), in step order: a stock that a dispatch, a grid burn or packaging left in [-b, 0) and that was
    # set to 0, with b = 1e-12 max(A-bar, |q|, draw, 2^-1022) (Q58 M11; design §12 "Stock clamps and dust lots"), value
    # < 0; and a queue lot with 0 < qty <= 1e-12 that left the book, at its (chokepoint, k) slot, value its qty > 0
    clamps: tuple[tuple[int, float], ...] = ()
    invalid: tuple[str, ...] = ()  # entries dropped by the validity rules, logged (§9.3)
    # (action slot, value), in slot order: an executed request on an edge with a subnormal factor (4) that the edge
    # clamp lowered to the edge's residual u'_e, value < 0 the residual it would have left, within the band of
    # ``clamps`` with q the request (Q95; ``clip.edge_clamp``)
    edge_clamps: tuple[tuple[int, float], ...] = ()
    # (override slot, value), per chokepoint in slot order: an override on an out-edge with a subnormal factor (4) that
    # the edge clamp lowered to the out-edge's residual u'_e before (6), value < 0 the residual it would have left,
    # within the band of ``edge_clamps`` with q the override request (Q95; ``chokepoint._overrides``)
    override_clamps: tuple[tuple[int, float], ...] = ()

    def as_json(self) -> dict:
        """JSON-able form with sorted keys and floats by repr (for the trajectory hash).

        ``edge_clamps`` and ``override_clamps`` enter only when a clamp was logged in them, so a record without one
        hashes as before the fields existed (Q95).
        """

        def enc(v):
            if isinstance(v, np.ndarray):
                return v.tolist()  # float64 arrays: Python floats, by repr in the JSON
            if isinstance(v, dict):
                return [[list(k) if isinstance(k, tuple) else k, float(val)] for k, val in sorted(v.items(), key=_key)]
            if isinstance(v, CostComponents):
                return v.as_dict()
            if isinstance(v, tuple):
                return [list(x) if isinstance(x, tuple) else x for x in v]
            return v

        return {
            name: enc(getattr(self, name))
            for name in self.__dataclass_fields__
            if name not in _WHEN_LOGGED or getattr(self, name)
        }


def _key(item):
    k = item[0]
    return tuple(-1 if x is None else x for x in k) if isinstance(k, tuple) else (k,)


@dataclass
class Trajectory:
    """One episode: actions as received, the weekly records and the terminal credit (23)-(25)."""

    instance_hash: str
    omega_hash: str
    policy: str
    regime: str
    actions: list[dict] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)
    salvage: float | None = None  # S_T in USD at T (23)
    salvage_cents: int | None = None  # S^¢_T (24)
    instance_digest: str = ""  # content digest of the instance simulated (V3)
    marks_digest: str = ""  # content digest of the marks simulated (V3)
    # "week t (ExceptionType)" once a step raised: ``actions`` then ends with week t's action, and ``records`` (and the
    # terminal credit) hold the completed weeks only, wherever in the step it failed (§12 'Failed steps'; DET-P3-5)
    failed: str | None = None
    # (week, code of information.wire.WIRE_FAILURES) of every week the runner stepped with a whole-week wire failure
    # (``Env.step(..., wire_failure=code)``, M3): the runner's cause, which no reply value can forge; hashed only when
    # set, so an in-process trajectory hashes as before the field existed, and replay passes it back (design §12 M3)
    wire_failures: list[tuple[int, str]] = field(default_factory=list)

    @property
    def J_cents(self) -> int:
        """J^¢ = sum_t C^¢_t - S^¢_T (24); requires a finished episode."""
        if self.salvage_cents is None:
            raise ValueError("episode not finished")
        return sum(r.cost_cents for r in self.records) - self.salvage_cents

    def rewards_cents(self) -> list[int]:
        """r_t of (25): -C^¢_t, and -C^¢_T + S^¢_T at T; they sum to -J^¢ exactly."""
        r = [-rec.cost_cents for rec in self.records]
        if self.salvage_cents is not None and r:
            r[-1] += self.salvage_cents
        return r

    def sha256(self) -> str:
        """SHA-256 of the canonical JSON (§2.3) of the actions, records and terminal credit (V1, B2).

        A failed trajectory adds its ``failed`` mark, and one with wire failures its ``wire_failures``; a trajectory
        without them hashes as before they existed.
        """
        doc = {
            "instance_hash": self.instance_hash,
            "omega_hash": self.omega_hash,
            "policy": self.policy,
            "regime": self.regime,
            "actions": self.actions,
            "records": [r.as_json() for r in self.records],
            "salvage": self.salvage,
            "salvage_cents": self.salvage_cents,
            "instance_digest": self.instance_digest,
            "marks_digest": self.marks_digest,
        }
        if self.failed is not None:
            doc["failed"] = self.failed
        if self.wire_failures:
            doc["wire_failures"] = [[int(w), str(code)] for w, code in self.wire_failures]
        return sha256_hex(doc)
```

### `shockbench_flow/instance/schema.py`

```python
"""Frozen dataclasses of an instance (design §2.1, §2.3, §2.4; Q58, Q70, Q79, Q85).

An instance is one canonical JSON file (`instance/io.py`), loaded into the dataclasses below. Integer codes follow
design §4.1 (Q87): regions, nodes, edges, lanes and commodities are indexed by their position in the file's lists, and
every cross-reference below is such an index. Chokepoints, grids, fabs, OSATs and sink demands also have *ordinals*,
their order of appearance among the nodes (``Instance.chokepoints`` etc.); arrays over chokepoints (marks, the omega
container's ``wr_class``) are indexed by that ordinal.

Money is in USD per unit here; costs become integer cents only in the cost function (24).
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from shockbench_flow.digest import semantic_digest


SCHEMA_VERSION = "1"

# Vocabularies, frozen with schema_version (design §9.2; Q86)
NODE_TYPES = ("source", "terminal", "grid", "chokepoint", "material", "fab", "osat", "sink")
MODES = ("sea", "air", "pipeline", "grid")
POOLS = ("tb", "ct")  # tanker-and-bulk (energy, GWh) and container (semiconductor goods, wafer-eq) (Q49, Q76)
WAR_RISK_CLASSES = ("none", "red_sea", "hormuz_2026")  # codes 0, 1, 2 of the omega container's wr_class
PRIORITIES = ("base_first", "proportional", "industrial_first")  # energy priority pri_g (18) (Q57)
FAB_CLASSES = ("leading", "mature", "memory")
SUPPLY_TYPES = ("source", "material")  # V^sup (Q79)
UNMODELLED = "unmodelled"  # JSON key of the unmodelled generation segment k = ∅ in (15), (17), (18)
WEEKS_PER_YEAR = 52  # length of a seasonal profile m^sea (§2.1)
# instance kinds, whose index is the instance-kind code of the seed keys (design §4.1; Q87); defined here, next to the
# other vocabularies, so that the loader checks `kind` without importing omega (omega/codes.py re-exports it)
INSTANCE_KINDS = ("tiny", "small", "full", "abstract")
FLAGSHIP_KINDS = ("small", "full")  # the sizes built from spec §4.6-4.7 (design §2.2; Q83)


class FrozenDict(dict):
    """An immutable dict for the mapping fields of frozen dataclasses.

    Unlike ``MappingProxyType`` it pickles, so instances, marks and environment snapshots cross process boundaries
    (joblib workers, resume, B2).
    """

    def _readonly(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __reduce__(self):
        return (FrozenDict, (dict(self),))

    def __hash__(self):  # frozen dataclasses holding one stay hashable when their values are
        return hash(tuple(sorted(self.items(), key=repr)))


def frozen_map(d: Mapping) -> Mapping:
    """An immutable (and picklable) copy of a dict, for mapping fields of frozen dataclasses."""
    return FrozenDict(d)


@dataclass(frozen=True)
class Provenance:
    """Tag and source of one leaf parameter (design conventions; V20, V23)."""

    tag: str  # REAL, DERIVED(...), CALIBRATED(...), SYNTHETIC(prior...), SYNTHETIC, SYNTHETIC(placeholder)
    source: str
    lo: float | None = None
    hi: float | None = None

    @property
    def placeholder(self) -> bool:
        return self.tag.startswith("SYNTHETIC(placeholder")


@dataclass(frozen=True)
class Commodity:
    """A commodity of §2.1; ``override`` marks tanker cargo K^ov (Q34)."""

    id: str
    unit: str
    pool: str  # "tb" or "ct"
    v: float  # customs value v_k, USD per unit
    override: bool
    disposal_cost: float  # c^disp_k, USD per unit (Q24)

    @property
    def pool_index(self) -> int:
        return POOLS.index(self.pool)


@dataclass(frozen=True)
class StockSlot:
    """One stock I_ik of (8): a (node, commodity) pair the node can hold.

    At a chokepoint the slot is the queue: ``storage`` is None (no cap, Q58 M4), ``holding`` is unused (queue holding
    h^Q by war-risk class is in ``ChokepointAttrs``) and ``salvage`` is the queue's nu = c^min x the commodity's share,
    valued like every other node (Q93).
    """

    node: int
    k: int
    storage: float | None  # I^max_ik; None only at chokepoints
    holding: float  # h_ik, USD per unit-week; 0 at supply nodes (Q79)
    salvage: float  # nu_ik, USD per unit; 0 at supply nodes (Q58 E18, Q79), c^min x share elsewhere (chokepoints: Q93)
    supply: float = 0.0  # availability varsigma-bar^0_ik per week at supply nodes (Q79); 0 elsewhere


@dataclass(frozen=True)
class ChokepointAttrs:
    """Chokepoint attributes (§2.1, §3.4): throughput kappa_cb = k_c mu_cb o^t_c per pool (9)."""

    mu: tuple[float, float]  # normal traffic mu_cb per pool, (tb, ct)
    k_c: float  # clearance ratio (Q7)
    cls: str  # chokepoint class (free text in v1)
    queue_holding: Mapping[int, tuple[float, float, float]]  # k -> h^Q_ck by war-risk class (none, red_sea, hormuz)
    war_risk_cost: Mapping[int, tuple[float, float, float]]  # k -> per-transit cost c-bar^wr_k by class (11)

    @property
    def kappa0(self) -> tuple[float, float]:
        """Nominal throughput k_c mu_cb per pool (tb, ct), the factor of o^t_c in (9)."""
        return (self.k_c * self.mu[0], self.k_c * self.mu[1])


@dataclass(frozen=True)
class GridAttrs:
    """Grid attributes (§2.1, §3.5): segments (15), buffer (16), allocation (17)-(18)."""

    base_load: float  # y-bar_g, GWh/wk
    deliverable: float  # G-bar^0_g, GWh/wk
    shares: Mapping[int | None, float]  # zeta_gk by fuel commodity index; None = the unmodelled segment ∅
    days_cover: Mapping[int, float]  # D_g per fuel, days
    ibar: Mapping[int, float]  # reference buffer I-bar_g per fuel, GWh (16), as the instance declares it
    rationed: int | None  # commodity index of the gas segment, the only one rationed (Q27); None if no gas
    voll: float  # VOLL_g, USD/GWh (Q28)
    priority: str  # pri_g, one of PRIORITIES (Q57)

    @property
    def fuels(self) -> tuple[int, ...]:
        """K^E_g: the fuel commodities of the grid, in commodity order."""
        return tuple(sorted(k for k in self.shares if k is not None))


@dataclass(frozen=True)
class FabAttrs:
    """Fab attributes (§2.1, §3.5): (12)-(14)."""

    cap0: float  # wafers per week
    e: float  # GWh per wafer; 0 without a grid
    grid: int | None  # node index of the fab's grid, or None (energy outside the model)
    cls: str  # leading, mature or memory
    input: int  # commodity index of the wafer input
    product: int  # commodity index of the raw chip k^raw(f)
    tau: int  # tau^fab_f, weeks
    w_scr: int  # scrap window w^scr_f, 1 <= w_scr <= tau (14)


@dataclass(frozen=True)
class OsatAttrs:
    """OSAT attributes (§2.1, §3.5): (19)."""

    thr: float  # throughput thr_i per week over all packaged commodities
    tau: int  # tau^osat, weeks
    packages: Mapping[int, int]  # raw chip commodity -> packaged chip commodity


@dataclass(frozen=True)
class TerminalAttrs:
    """Terminal attributes (§2.1); the throughput enters as the capacity of the terminal's grid edge (§2.2)."""

    throughput: float


@dataclass(frozen=True)
class Node:
    """A node of §2.1 with its type-specific block (exactly one is set for chokepoints, grids, fabs, OSATs)."""

    id: str
    type: str  # one of NODE_TYPES
    region: int  # region index
    chokepoint: ChokepointAttrs | None = None
    grid: GridAttrs | None = None
    fab: FabAttrs | None = None
    osat: OsatAttrs | None = None
    terminal: TerminalAttrs | None = None


@dataclass(frozen=True)
class AltRef:
    """A tagged `alt_of` reference, ``{"lane": id}`` or ``{"edge": id}`` (Q85)."""

    kind: str  # "lane" or "edge"
    index: int


@dataclass(frozen=True)
class Edge:
    """An edge of §2.1; coupling edges (mode grid) carry no commodity and accept no request (Q58 M5)."""

    id: str
    tail: int
    head: int
    mode: str
    K: tuple[int, ...]  # permitted commodities K_e; empty on coupling edges
    tau: int  # lead time in weeks (fixed in v1, Q87)
    c0: float  # unit freight c^0_e, USD
    u0: float | None  # capacity u^0_e per week; None on coupling edges
    alt_of: AltRef | None
    pool: str | None  # one pool per edge (§2.1 proposal); None on coupling edges

    @property
    def coupling(self) -> bool:
        return self.mode == "grid"


@dataclass(frozen=True)
class Lane:
    """A lane l = (i, c_1, ..., c_r, j): a path whose interior nodes are chokepoints (Q4)."""

    id: str
    edges: tuple[int, ...]
    chokepoints: tuple[int, ...]  # interior node indices, in path order
    alt_of: AltRef | None


@dataclass(frozen=True)
class Demand:
    """Demand of packaged chip k at sink i, (20)-(22)."""

    node: int
    k: int
    dbar: float  # base demand d-bar_ik = upsilon F-bar_ik (22); the lognormal's median (§12)
    pi: float  # shortage penalty pi_ik, USD per unit (Q85)
    backlog: bool  # True: backlog sink; False: lost sales (20)
    phi: float  # AR(1) coefficient phi^d of (21)
    sigma: float  # innovation SD sigma^d of (21)
    seasonal: tuple[float, ...] | None  # m^sea_ik by week of year (52 values), None = identically 1
    shock: tuple[float, float, float]  # Xi_ik by the sink region's conflict layer (none, minor, war)

    def m_sea(self, t: int) -> float:
        """m^sea(t) of (21) for episode week t: ``seasonal[(t - 1) % 52]``, 1 without a profile."""
        return 1.0 if self.seasonal is None else self.seasonal[(t - 1) % WEEKS_PER_YEAR]


@dataclass(frozen=True)
class InitialShipment:
    """A pipeline shipment x^{t'<=0}_ek of the initial state (2), carrying its lane when it rides one."""

    edge: int
    k: int
    lane: int | None
    qty: float
    dispatch_week: int  # t' <= 0
    arrival_week: int  # t' + tau_e >= 1


@dataclass(frozen=True)
class InitialWip:
    """Gross work in process maturing at ``out_week`` (fab: raw chips; OSAT: packaged chips)."""

    node: int
    k: int  # the output commodity
    qty: float
    out_week: int  # 1 <= out_week <= tau


@dataclass(frozen=True)
class InitialLot:
    """A queue lot at a chokepoint at reset (§3.4 lot fields; Q58 M13)."""

    chokepoint: int  # node index
    k: int
    qty: float
    lane: int
    next_edge: int
    dispatch_week: int
    entry_edge: int
    arrival_week: int  # <= 0


@dataclass(frozen=True)
class InitialState:
    """The declared initial state (§2.3, §2.4): stock, pipeline, WIP and queue lots."""

    stock: tuple[tuple[int, int, float], ...]  # (node, k, qty) for every nonzero stock, supply nodes included
    pipeline: tuple[InitialShipment, ...]
    fab_wip: tuple[InitialWip, ...]
    osat_wip: tuple[InitialWip, ...]
    queue_lots: tuple[InitialLot, ...]


@dataclass(frozen=True)
class Params:
    """Instance-wide parameters (§2.4, §3.3, §3.5)."""

    psi: float  # rationing threshold psi (Q36)
    alpha_max: float  # overproduction ceiling alpha_max (13)
    tau_alpha: float  # ceiling lag tau_alpha in weeks (13)
    upsilon: float  # utilisation (22) (Q76 owner D2)
    fleet_share: tuple[float, float]  # s^b_fl per pool (tb, ct) (7)
    fleet_measure: tuple[float, float]  # F^b per pool (tb, ct), GWh-wk and wafer-eq-wk (7); declared, checked (§3.3)
    top_tariff: float  # the instance's top tariff rate, which sets pi (Q85)
    forecast_shares: tuple[float, ...]  # w^f_j, j = 0..8 (48)


@dataclass(frozen=True)
class Instance:
    """A loaded instance (§2.3). Build it with ``instance.io.load_instance``; never mutate it."""

    schema_version: str
    instance_id: str
    kind: str  # tiny, small, full or abstract (instance-kind code of §4.1)
    T: int
    units: Mapping[str, str]
    regions: tuple[str, ...]
    region_class: tuple[str, ...]  # per region (Q59)
    commodities: tuple[Commodity, ...]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    lanes: tuple[Lane, ...]
    stock_slots: tuple[StockSlot, ...]
    demands: tuple[Demand, ...]
    routing_table: tuple[tuple[int, int, tuple[int, ...]], ...]  # (origin region, dest region, lanes)
    compatibility: tuple[tuple[int, int], ...]  # (source node, terminal node)
    use: tuple[tuple[int, int], ...]  # (material node, fab node)
    chokepoint_adjacency: Mapping[int, tuple[int, ...]]  # chokepoint node -> adjacent regions (Q58)
    trade_adjacency: tuple[tuple[int, int, float], ...]  # (region, region, weight) entries of A (R3 §3)
    initial_state: InitialState
    prohibitions_at_reset: tuple[tuple[int, int], ...]  # (edge, k) pairs of Z_0, in force all episode
    params: Params
    provenance: Mapping[str, Provenance]
    hash: str  # SHA-256 of the canonical JSON (§2.3)
    raw: Mapping = field(repr=False, compare=False)  # the parsed canonical JSON, shipped in Static (Q86)

    # ----- derived index tables, filled by the loader -------------------------------------------------------------
    node_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    edge_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    lane_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    commodity_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    region_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    out_edges: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # delta+(i)
    in_edges: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # delta-(i)
    chokepoints: tuple[int, ...] = field(repr=False, compare=False, default=())  # node indices, ordinal order
    grids: tuple[int, ...] = field(repr=False, compare=False, default=())
    fabs: tuple[int, ...] = field(repr=False, compare=False, default=())
    osats: tuple[int, ...] = field(repr=False, compare=False, default=())
    sinks: tuple[int, ...] = field(repr=False, compare=False, default=())
    supply_nodes: tuple[int, ...] = field(repr=False, compare=False, default=())
    slot_index: Mapping[tuple[int, int], int] = field(repr=False, compare=False, default_factory=dict)  # (i,k)->slot
    # the fleet-slack terms of (7), the one table of E^dup: (edge, lane or None, Delta tau). An edge-level sea
    # duplicate counts all its flow (lane None); a sea duplicate lane is charged its extra transit over the route it
    # replaces, counted from where it diverges, on the edge where it diverges: on its entry edge, the flow dispatched
    # on that lane; on a turn-back out of a chokepoint, all the flow of that edge (design §3.3: "Cape, Lombok-Sunda
    # and east-of-Taiwan lanes and turn-back edges into them")
    dup_items: tuple[tuple[int, int | None, int], ...] = field(repr=False, compare=False, default=())
    action_slots: tuple[tuple[int, int, int | None], ...] = field(repr=False, compare=False, default=())
    override_slots: tuple[tuple[int, int, int, int | None], ...] = field(repr=False, compare=False, default=())
    action_slot_index: Mapping[tuple, int] = field(repr=False, compare=False, default_factory=dict)  # slot -> index
    override_slot_index: Mapping[tuple, int] = field(repr=False, compare=False, default_factory=dict)
    # ordinals of chokepoints, grids, fabs and OSATs by node index
    chokepoint_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    grid_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    fab_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    osat_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    commodity_pool: tuple[int, ...] = field(repr=False, compare=False, default=())  # pool index per commodity
    grid_fabs: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # fab ordinals per grid ord.
    lane_K: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # K on every edge, per lane
    # (lane, chokepoint node) -> (edge into c, edge out of c = e_l(c)) of (52), for every interior chokepoint
    lane_through: Mapping[tuple[int, int], tuple[int, int]] = field(repr=False, compare=False, default_factory=dict)
    # (chokepoint node, out-edge) -> the lanes the out-edge continues, in lane order (out-edges on some lane only)
    lanes_continuing: Mapping[tuple[int, int], tuple[int, ...]] = field(repr=False, compare=False, default_factory=dict)
    # out-edges of chokepoints that continue some lane (they carry the war-risk transit cost (11))
    continuing_edges: frozenset[int] = field(repr=False, compare=False, default=frozenset())
    # the warm start per rung of §2.3 (owner queue M5-O37 (b)): the γ rung whose block ``initial_state.stock`` holds
    # (None: one start for every rung, `tiny` and a point-mass build) and every rung's block, that one included, as
    # (node, k, qty) entries; outside the content digest, which covers the block in use (``at_rung`` swaps it in)
    start_rung: float | None = field(repr=False, compare=False, default=None)
    rung_stock: Mapping[float, tuple[tuple[int, int, float], ...]] = field(
        repr=False, compare=False, default_factory=dict
    )
    # week-invariant tables of other modules (simulator, cost, observation), built on first use by ``memo``; not an
    # init field, so ``dataclasses.replace`` starts with an empty memo and never carries tables of another instance
    _memo: dict = field(init=False, repr=False, compare=False, default_factory=dict)

    def memo(self, key: str, build: Callable[["Instance"], Any]) -> Any:
        """The table ``build(self)``, built once per instance object and kept under ``key``."""
        if key not in self._memo:
            self._memo[key] = build(self)
        return self._memo[key]

    @property
    def content_digest(self) -> str:
        """SHA-256 of the instance's content (every compared field), the V3 identity the simulator and LP check."""
        return self.memo("content_digest", semantic_digest)

    @property
    def rungs(self) -> tuple[float, ...]:
        """The γ rungs with a stock block of their own, ascending; empty with one start for every rung (§2.3)."""
        return tuple(sorted(self.rung_stock))

    def at_rung(self, gamma: float) -> "Instance":
        """The instance at γ rung ``gamma``: that rung's block as ``initial_state.stock`` (§2.3; M5-O37 (b)).

        The file's hash label and raw JSON, its own content digest; everything but the on-hand stock is shared. The
        instance itself at its own rung and on an instance with one start for every rung (`tiny`, a point-mass build),
        so those never move; built once per instance and rung (``memo``).

        Raises:
            ValueError: if the instance keeps a block per rung and none for ``gamma``.

        """
        if not self.rung_stock or gamma == self.start_rung:
            return self
        if gamma not in self.rung_stock:
            raise ValueError(f"{self.instance_id} has no warm start at rung {gamma!r} (its rungs: {self.rungs}; §2.3)")
        return self.memo(f"at_rung:{gamma!r}", lambda inst: _at_rung(inst, gamma))

    def at_digest(self, digest: str) -> "Instance":
        """The instance at the rung whose content digest is ``digest`` (V3); the instance itself if none is.

        What an omega's ``meta_instance_digest`` or a record's ``instance_digest`` names: the episode's block.
        """
        if self.content_digest == digest:
            return self
        return next((r for r in map(self.at_rung, self.rungs) if r.content_digest == digest), self)

    @property
    def family_digest(self) -> str:
        """The content digest at the file's own rung: what naive's F_Q and the D9 fallback are a function of.

        F_Q reads no stock (§12 'Warm start per rung'), so the instances of one file at every rung share it; it is the
        content digest itself on an instance with one start and on the file as loaded.
        """
        return self.memo("family_digest", _family_digest)

    def ordinal(self, kind: str, node: int) -> int:
        """Ordinal of a node among ``kind`` ∈ {chokepoints, grids, fabs, osats}."""
        return getattr(self, _ORDINAL_FIELD[kind])[node]

    def lane_next_edge(self, lane: int, edge: int) -> int | None:
        """The lane's edge after ``edge`` (e_l(c) when ``edge`` enters chokepoint c), or None at the lane's end.

        None also when ``edge`` is not on the lane.
        """
        through = self.lane_through.get((lane, self.edges[edge].head))
        return through[1] if through is not None and through[0] == edge else None

    def lane_destination(self, lane: int) -> int:
        """Head node of the lane's last edge."""
        return self.edges[self.lanes[lane].edges[-1]].head

    def continues_lane(self, edge: int) -> bool:
        """True if ``edge`` leaves a chokepoint as the continuation of some lane (war-risk transit cost, (11))."""
        return edge in self.continuing_edges


def _at_rung(inst: Instance, gamma: float) -> Instance:
    """``inst`` with the block of rung ``gamma`` as its initial stock (a fresh memo; ``Instance.at_rung``)."""
    start = dataclasses.replace(inst.initial_state, stock=inst.rung_stock[gamma])
    return dataclasses.replace(inst, initial_state=start, start_rung=gamma)


def _family_digest(inst: Instance) -> str:
    """``Instance.family_digest``: the content digest of ``inst`` at its file's own rung (``raw`` names it)."""
    own = inst.raw["initial_state"].get("rung") if inst.rung_stock else None
    return inst.content_digest if own is None else inst.at_rung(own).content_digest


_ORDINAL_FIELD = {
    "chokepoints": "chokepoint_ordinal",
    "grids": "grid_ordinal",
    "fabs": "fab_ordinal",
    "osats": "osat_ordinal",
}
```
