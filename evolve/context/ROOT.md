ShockBench-Flow is a Gymnasium control task. Every week of an episode an agent
decides how much of each good to send along each route of a supply network.
Disruptions (a strait closes, a route is sanctioned, a tariff jumps, a factory
goes down) are drawn before the episode starts and nothing the agent does
changes them. The agent sees the network as it is this week, its stock and
shipments, a demand forecast, and noisy early warnings and announcements. An
episode costs money (USD); lower is better.

The score compares an agent's cost with two references on the same scenarios:
**0** is the naive rule (keep shipping the normal plan), **1** is the
clairvoyant plan (it knew every disruption in advance); below 0 is worse than
naive. The package and the board call it RSS.

---

## The task

An episode is T weeks (26 on Tiny, 52 on Small, 104 on Full). Each week your
agent chooses how much of each commodity to send along each route (an _action
slot_: an edge, or the first edge of a sea lane through one or more straits),
and, for tanker cargo waiting at a strait, whether it leaves by the default
rule, by your own quantities, or waits. The environment then clips your orders
to what is in stock and what the routes can carry this week, moves the goods,
runs the factories and power grids, serves demand and charges the week's cost.

The cost J of an episode is the sum over weeks of freight, war-risk surcharges,
tariffs, holding (higher for cargo queued at a strait), a penalty for every unit
of demand not served, disposal and power shed at the grids, minus the value of
what is left at the end. Lower is better. The disruptions of an episode
(closures, sanctions, tariffs, conflicts, factory outages) are drawn before it
starts from a public generator; nothing your agent does changes them.

**What your agent sees: the `standard` regime**, the one the leaderboards use.
Besides the network as it is this week, your own state and the demand forecast,
it gives three early signals of disruptions that have not acted yet:

- `warning.score`: an early-warning score per region, pair of rival regions and
  strait, with a one-week lag;
- `messages.*`: announcement threads (tariff proposals and final notices,
  sanction threats, military threats); some are false alarms that never take
  effect;
- `pending_prohibitions.*`: announced sanctions not yet in force, with the week
  each takes effect.

The naive rule that anchors the score sees none of these and ignores
disruptions, so using them well is where an agent can gain.

## The interface

```python
class Agent:
    def __init__(self, config=None):  # once per episode
        ...

    def act(self, observation):       # once per week
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
```

- `config` is a dict: `static` (the network's tables: `nodes`, `edges`, `lanes`,
  `commodities`, `action_slots`, `override_slots`, `sinks`, and the whole public
  instance under `static["instance"]`), `regime`, `T`, `policy_seed` (seed your
  random generators with it), `layout` (what each position of a dense
  observation block stands for), `release_modes` and `spaces` (every array's
  shape and dtype). Nothing hidden is in it.
- `observation` is a dict of numpy arrays with fixed shapes, keyed by strings
  (`observation["stock.qty"]`). Every field `x` comes with `x.observed`, 1 where
  the value is shown this week and 0 where it is hidden or padding. Lists of
  varying length (shipments in transit, messages) are padded to a fixed size.
- The action is a dict: `flows` (one quantity per action slot, 0 or more),
  `override_qty` (one per override slot) and `release_mode` (per strait and
  tanker commodity: 0 the default release, 1 your `override_qty`, 2 hold). The
  last two may be left out.
- `action_mask` is 1 on every slot that may carry goods this week (no sanction
  on its route). It does not show closures or capacity: read `graph_now.open`
  (how open each strait is, 1 to 0) and `graph_now.u` (each edge's capacity this
  week).
- The nominal weekly flows (the normal plan) are the week-0 shipments of
  `static["instance"]["initial_state"]["pipeline"]`.

Every field, with its shape, dtype, index set and meaning, is in
[fields/tiny.md](fields/tiny.md), [fields/small.md](fields/small.md) and
[fields/full.md](fields/full.md), generated from the installed package by
`uv run python scripts/fields_docs.py`. The fields that matter most at first:

| key                                           | what it is                                                                                                  |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `week`                                        | the week to decide, 1 to T                                                                                  |
| `stock.qty`                                   | stock on hand per (node, commodity), rows `layout["stock_slots"]`                                           |
| `backlog.qty`                                 | unserved demand carried at each market                                                                      |
| `graph_now.u`, `graph_now.c`, `graph_now.tau` | each edge's capacity, freight cost and lead time this week                                                  |
| `graph_now.open`                              | each strait's open fraction, rows `layout["chokepoints"]`                                                   |
| `graph_now.prohibited`, `graph_now.tariff`    | sanctions and tariffs per (edge, commodity)                                                                 |
| `demand_forecast.qty`                         | the demand forecast for the next 8 weeks                                                                    |
| `last_week.cost_components`                   | last week's cost by component (freight, war risk, tariff, holding, queue holding, shortage, disposal, shed) |
| `warning.score`                               | the early-warning scores (`standard` only)                                                                  |
| `action_mask`, `override_mask`                | the slots you may use this week                                                                             |

Under gymnasium your agent needs the `config` the server builds; `agent_config`
makes it from the reset:

```python
import gymnasium as gym
import shockbench_flow_gym
from shockbench_flow_agent import agent_config

env = gym.make("ShockBench/Small-v0")
obs, info = env.reset(options={"episode": 0})
agent = Agent(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
```

`gym.make` options: `regime` (`"standard"`, the scored one; `"prediction_free"`
hides the three signals), `dense_reward` (a shaped reward whose sum is still
minus the cost, up to a constant), `render_mode="rgb_array"`, `entropy` (the
scenarios' root: 0, the default, is the public dev root; any other integer below
2\*\*128 is a training root of your own) and `gamma` (the disruption intensity:
0.62, the default and the scored one, 0.79, 0.95, 0.97). The gymnasium
environment plays no fallback: an exception in `act` stops your script.

## Small and Full

Small (the public board's) and Full (the private board's) have the same keys as
Tiny except two blocks, stored compactly:

- `pipeline.*` is grouped: one entry per (edge, commodity, lane, arrival week),
  its `qty` the total of the shipments.
- The cargo queued at the straits is one dense array, `queue_lots.qty`, of shape
  (number of lot keys, T). Row `i` is `config["layout"]["lot_keys"][i]`, a
  (strait node, commodity, lane, next edge) tuple; column `w - 1` holds the
  quantity that reached the strait in week `w` and still waits.
- Tiny's per-lot lists (`queue_lots.lot_id`, `.chokepoint`, `.k`, ...) do not
  exist there: an agent that reads them raises `KeyError` every week. Test
  `"lot_keys" in config["layout"]` to tell the layouts apart.

Read every shape from `config["spaces"]`, never from Tiny's tables.

## Common mistakes

- **Indexing the observation by position.** `observation[2]` raises `KeyError`:
  it is a dict keyed by strings.
- **`__init__` without `config`.** The server calls `Agent(config)`;
  `def __init__(self)` raises, and naive plays the whole episode.
- **Returning the wrong thing.** `act` returns a dict with at least `flows`, a
  float array with one entry per action slot (20, 108 or 395); `release_mode` is
  an array, not a scalar.
- **Flows on sanctioned routes.** Entries on a prohibited slot, and negative or
  non-finite quantities, are dropped (the rest of the action stands). Multiply
  `flows` by `observation["action_mask"]`.
- **Trusting `action_mask` for closures.** What you send into a closed strait
  waits in its queue: check `graph_now.open`.
- **Imports the server lacks.** Only the standard library, numpy, scipy and
  torch exist there. `sbf check` fails an agent that imports anything else.
- **Files next to `agent.py`.** Load them relative to it:
  `Path(__file__).parent / "weights.npz"`.

---

Generated for instance `chokepoint-tiny` (T = 26, regime `standard`) by `uv run python scripts/fields_docs.py`.

**Observation** (`act(observation)`): a dict of numpy arrays. Every key below except `action_mask.observed` and `override_mask.observed` is followed by `<key>.observed`, an int8 array of the same shape, 1 where the value is present and 0 where it is unobserved or padding (the value is then 0).

| key | shape | dtype | indexed by | meaning |
| --- | --- | --- | --- | --- |
| `week` | (1,) | int64 | - | t, the week to decide (1-based); every other field is the state at instant t - 1 |
| `stock.qty` | (15,) | float64 | layout.stock_slots | on-hand stock I^{t-1} per (node, k), chokepoints excluded (see queue_lots) |
| `backlog.qty` | (1,) | float64 | layout.demands | unserved demand carried at each backlog sink (0 at lost-sales sinks) |
| `pipeline.edge` | (256,) | int64 | padded list | shipments in transit: the edge they travel |
| `pipeline.k` | (256,) | int64 | padded list | commodity index |
| `pipeline.lane` | (256,) | int64 | padded list | lane index; unobserved for a shipment off any lane |
| `pipeline.qty` | (256,) | float64 | padded list | quantity; its observed-mask marks the live entries of the list |
| `pipeline.arrival_week` | (256,) | int64 | padded list | week the shipment reaches the edge's head |
| `queue_lots.lot_id` | (128,) | int64 | padded list | lots waiting at a chokepoint (FIFO book): lot id |
| `queue_lots.chokepoint` | (128,) | int64 | padded list | chokepoint node |
| `queue_lots.k` | (128,) | int64 | padded list | commodity index |
| `queue_lots.qty` | (128,) | float64 | padded list | quantity |
| `queue_lots.lane` | (128,) | int64 | padded list | lane the lot follows |
| `queue_lots.next_edge` | (128,) | int64 | padded list | edge the lot leaves the chokepoint by |
| `queue_lots.arrival_week` | (128,) | int64 | padded list | week the lot reached the chokepoint |
| `queue_lots.dispatch_week` | (128,) | int64 | padded list | week the lot was dispatched |
| `queue_lots.entry_edge` | (128,) | int64 | padded list | edge the lot entered the chokepoint by |
| `wip.node` | (128,) | int64 | padded list | work in process at fabs and OSATs (gross): node |
| `wip.k` | (128,) | int64 | padded list | output commodity |
| `wip.qty` | (128,) | float64 | padded list | quantity |
| `wip.out_week` | (128,) | int64 | padded list | week it becomes stock |
| `graph_now.u` | (25,) | float64 | edges | capacity u per edge this week; unobserved on grid couplings |
| `graph_now.c` | (25,) | float64 | edges | freight cost per unit per edge |
| `graph_now.tau` | (25,) | int64 | edges | lead time in weeks per edge |
| `graph_now.prohibited` | (25, 4) | int8 | edges x commodities | 1 where (edge, k) is prohibited (sanctions, export controls) |
| `graph_now.tariff` | (25, 4) | float64 | edges x commodities | tariff rate on (edge, k) |
| `graph_now.open` | (1,) | float64 | layout.chokepoints | open fraction o_c of each chokepoint (a strait, canal or the Cape route; 1 open, 0 closed) |
| `graph_now.kappa.tb` | (1,) | float64 | layout.chokepoints | throughput kappa_cb of the tanker/bulk pool |
| `graph_now.kappa.ct` | (1,) | float64 | layout.chokepoints | throughput kappa_cb of the container pool |
| `graph_now.war_risk` | (1,) | int64 | layout.chokepoints | war-risk class code: 0 none, 1 red_sea, 2 hormuz_2026 |
| `graph_now.supply.avail` | (4,) | float64 | layout.supply_slots | supply available at each source and material slot |
| `graph_now.fab.R` | (3,) | float64 | layout.fabs | restoration factor R_f of each fab |
| `graph_now.fab.alpha_bar` | (3,) | float64 | layout.fabs | power multiplier alpha-bar_f of each fab |
| `graph_now.fab.cap_eff` | (3,) | float64 | layout.fabs | effective wafer capacity of each fab |
| `graph_now.grid.G_bar` | (2,) | float64 | layout.grids | deliverable generation G-bar_g of each grid |
| `graph_now.grid.y_bar` | (2,) | float64 | layout.grids | base load y-bar_g of each grid |
| `graph_now.osat.R` | (1,) | float64 | layout.osats | restoration factor R^osat of each OSAT |
| `graph_now.osat.thr_eff` | (1,) | float64 | layout.osats | effective throughput thr R^osat of each OSAT |
| `slot_mask` | (20,) | int8 | action slots | 1 where an edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity this week (the wire's convention; see action_mask) |
| `last_week.clip.requested` | (20,) | float64 | action slots | flow you requested last week |
| `last_week.clip.executed` | (20,) | float64 | action slots | flow executed after the capacity clip |
| `last_week.cost_components` | (8,) | float64 | layout.cost_components | last week's cost by component, USD |
| `last_week.sinks.demand` | (1,) | float64 | layout.demands | last week's demand |
| `last_week.sinks.served` | (1,) | float64 | layout.demands | last week's demand served |
| `last_week.sinks.lost` | (1,) | float64 | layout.demands | last week's demand lost |
| `last_week.shed.qty` | (2,) | float64 | layout.grids | power shed at each grid last week |
| `demand_forecast.qty` | (1, 8) | float64 | layout.demands x h | demand forecast for weeks t + h, h = 0..7 |
| `warning.score` | (16,) | float64 | layout.warning_units | early-warning score S^t per region, dyad and chokepoint |
| `messages.msg_id` | (2304,) | int64 | padded list | live announcement threads (announced, not effective, not withdrawn): thread id |
| `messages.channel` | (2304,) | int64 | padded list | channel code: 0 tariff_formal, 1 tariff_informal, 2 tariff_final, 3 sanction_legal, 4 ties_threat, 5 mid_threat |
| `messages.kind` | (2304,) | int64 | padded list | message kind code: 0 proposal, 1 final_notice, 2 threat, 3 publication, 4 withdrawal |
| `messages.region` | (2304,) | int64 | padded list | region index |
| `messages.target_kind` | (2304,) | int64 | padded list | target kind code: 0 chokepoint, 1 edge, 2 node, 3 region |
| `messages.target` | (2304,) | int64 | padded list | target index |
| `messages.k` | (2304,) | int64 | padded list | commodity index; unobserved when the message names none |
| `messages.announced_week` | (2304,) | int64 | padded list | week announced |
| `messages.stated_effective_week` | (2304,) | int64 | padded list | stated effective week; unobserved when none is stated |
| `pending_prohibitions.edge` | (2304,) | int64 | padded list | announced prohibitions not yet in force: edge |
| `pending_prohibitions.k` | (2304,) | int64 | padded list | commodity index |
| `pending_prohibitions.effective_week` | (2304,) | int64 | padded list | week it takes effect |
| `closure_end.chokepoint` | (192,) | int64 | padded list | closures acting now: chokepoint |
| `closure_end.end_week` | (192,) | int64 | padded list | announced end week; unobserved when unknown |
| `action_mask` | (20,) | int8 | action slots | 1 where no edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity this week (the inverse of slot_mask); capacities and closures are not checked: read graph_now.u and graph_now.open |
| `action_mask.observed` | (1,) | int8 | - | 1 when this week's mask was observed (0 in a blackout week: all slots allowed) |
| `override_mask` | (4,) | int8 | override slots | 1 where the slot's own out edge is not prohibited for its commodity this week; release_mode 1 sends every override slot of its pair, and a slot at 0 is dropped whatever its override_qty (one invalid entry, no cost): the pair's other slots stand, and with none valid the default release stays on |
| `override_mask.observed` | (1,) | int8 | - | 1 when this week's override mask was observed (0 in a blackout week) |

**Action** (the return value of `act`): a dict of numpy arrays (`override_qty` and `release_mode` may be left out: zeros, the default release).

| key | shape | dtype | indexed by | meaning |
| --- | --- | --- | --- | --- |
| `flows` | (20,) | float64 | action slots | quantity to dispatch on each (edge, commodity, lane) slot; 0 sends nothing |
| `override_qty` | (4,) | float64 | override slots | tanker cargo to release on each override slot, read where release_mode is 1 |
| `release_mode` | (1,) | int64 | layout.release_pairs | per (chokepoint, tanker commodity): 0 default release, 1 override, 2 hold |

**Action slots** (`flows`, `action_mask`, `slot_mask`, `last_week.clip.*`):

| slot | edge | from | to | commodity | lane |
| --- | --- | --- | --- | --- | --- |
| 0 | E0 | src_gulf | chk | lng | L0 |
| 1 | E0 | src_gulf | chk | lng | L1 |
| 2 | E3 | src_gulf | grid_eu | lng | - |
| 3 | E5 | src_usau | grid_tw | lng | - |
| 4 | E6 | src_usau | grid_eu | lng | - |
| 5 | E7 | src_ru | grid_eu | lng | - |
| 6 | E8 | src_ru | chk | lng | L2 |
| 7 | E9 | mat_jp | fab_tw | wafer | - |
| 8 | E10 | mat_jp | fab_tw | wafer | - |
| 9 | E11 | mat_jp | fab_cn | wafer | - |
| 10 | E12 | mat_jp | fab_cn | wafer | - |
| 11 | E13 | mat_jp | fab_us | wafer | - |
| 12 | E14 | mat_jp | fab_us | wafer | - |
| 13 | E17 | fab_tw | osat_sea | chip_le_raw | - |
| 14 | E18 | fab_tw | osat_sea | chip_le_raw | - |
| 15 | E19 | fab_cn | osat_sea | chip_le_raw | - |
| 16 | E20 | fab_us | osat_sea | chip_le_raw | - |
| 17 | E21 | fab_us | osat_sea | chip_le_raw | - |
| 18 | E22 | osat_sea | chk | chip_le | L3 |
| 19 | E24 | osat_sea | sink_us | chip_le | - |

**Override slots** (`override_qty`, `override_mask`):

| slot | chokepoint | commodity | out edge | lane |
| --- | --- | --- | --- | --- |
| 0 | chk | lng | E1 | L0 |
| 1 | chk | lng | E1 | L2 |
| 2 | chk | lng | E2 | L1 |
| 3 | chk | lng | E4 | - |

**Layout tables** (`config['layout']`, the positions of the densified blocks):

| table | entries |
| --- | --- |
| `stock_slots` | 0: src_gulf/lng, 1: src_usau/lng, 2: src_ru/lng, 3: grid_tw/lng, 4: mat_jp/wafer, 5: grid_eu/lng, 6: fab_tw/wafer, 7: fab_tw/chip_le_raw, 8: fab_cn/wafer, 9: fab_cn/chip_le_raw, 10: fab_us/wafer, 11: fab_us/chip_le_raw, 12: osat_sea/chip_le_raw, 13: osat_sea/chip_le, 14: sink_us/chip_le |
| `supply_slots` | 0: src_gulf/lng, 1: src_usau/lng, 2: src_ru/lng, 3: mat_jp/wafer |
| `demands` | 0: sink_us/chip_le |
| `chokepoints` | 0: chk |
| `fabs` | 0: fab_tw, 1: fab_cn, 2: fab_us |
| `grids` | 0: grid_tw, 1: grid_eu |
| `osats` | 0: osat_sea |
| `warning_units` | 0: region TW, 1: region KR, 2: region JP, 3: region CN, 4: region US, 5: region EU, 6: region GULF, 7: region RU, 8: region AU, 9: region SEA, 10: region IN, 11: region UA, 12: region KZ, 13: region ROW, 14: dyad 0, 15: chokepoint chk |
| `cost_components` | 0: freight, 1: war_risk, 2: tariff, 3: holding, 4: queue_holding, 5: shortage, 6: disposal, 7: shed |
| `release_pairs` | 0: chk/lng |

**Static tables** (`config['static']`, the episode's public tables): a table is a dict of equal-length lists, one entry per node, edge, lane, commodity, slot or sink, and the indices above point into them.

| field | here | meaning |
| --- | --- | --- |
| `instance` | keys `T`, `chokepoint_adjacency`, `commodities`, `compatibility`, `edges`, `initial_state`, `instance_id`, `kind`, `lanes`, `nodes`, `params`, `prohibitions_at_reset`, `provenance`, `region_class`, `regions`, `routing_table`, `schema_version`, `trade_adjacency`, `units`, `use` | the full public instance JSON; `initial_state.pipeline` holds the week-0 shipments of the nominal plan (the nominal flows the heuristic sample reads); edges there call the lead time `tau` |
| `instance_id` | `chokepoint-tiny` | the instance's name |
| `instance_hash` | `f0674842c1ec...` | SHA-256 of the instance |
| `T` | `26` | the horizon, in weeks |
| `units` | chip_le: wafer-eq 300 mm, chip_le_raw: wafer-eq 300 mm, cost: USD, lng: GWh fuel, wafer: wafer-eq 300 mm | the unit of each commodity's quantities, and of costs |
| `regions` | 14 entries | region names: `nodes.region`, `messages.region`, `dyads.*` and the warning's region units index them |
| `nodes.id` | 12 entries | node name |
| `nodes.type` | 12 entries | source, terminal, grid, chokepoint (a strait, canal or the Cape route), material, fab, osat or sink |
| `nodes.region` | 12 entries | region index |
| `commodities.id` | 4 entries | commodity name |
| `commodities.v` | 4 entries | customs value v_k, USD per unit |
| `commodities.pool` | 4 entries | chokepoint throughput pool: tb (tanker/bulk, `graph_now.kappa.tb`) or ct (container) |
| `commodities.override` | 4 entries | true for a tanker commodity, whose cargo queued at a chokepoint `release_mode` and `override_qty` steer |
| `edges.id` | 25 entries | edge name |
| `edges.tail` | 25 entries | node index the edge leaves |
| `edges.head` | 25 entries | node index the edge reaches |
| `edges.mode` | 25 entries | sea, air, pipeline or grid |
| `edges.tau0` | 25 entries | nominal lead time in weeks (`graph_now.tau` is this week's) |
| `edges.c0` | 25 entries | nominal freight, USD per unit (`graph_now.c` is this week's) |
| `edges.u0` | 25 entries | nominal capacity per week, null on a grid coupling (`graph_now.u` is this week's) |
| `edges.K` | 25 entries | commodity indices the edge may carry (empty on a grid coupling) |
| `edges.alt_of` | 25 entries | the route the edge duplicates, {'lane': i} or {'edge': i}, or null |
| `edges.pool` | 25 entries | chokepoint pool of its traffic, null on a grid coupling |
| `lanes.id` | 4 entries | lane name (a route through one or more chokepoints) |
| `lanes.edges` | 4 entries | edge indices along the lane, in order |
| `lanes.chokepoints` | 4 entries | chokepoint node indices it passes, in order |
| `lanes.alt_of` | 4 entries | the route the lane duplicates, {'lane': i} or {'edge': i}, or null |
| `action_slots.edge` | 20 entries | each slot's edge (on a lane, the lane's first edge) |
| `action_slots.k` | 20 entries | each slot's commodity index |
| `action_slots.lane` | 20 entries | each slot's lane index, null off any lane |
| `override_slots.chokepoint` | 4 entries | chokepoint node index |
| `override_slots.k` | 4 entries | tanker commodity index |
| `override_slots.out_edge` | 4 entries | edge the released cargo leaves the chokepoint by |
| `override_slots.lane` | 4 entries | lane index, null off any lane |
| `sinks.node` | 1 entry | demand node index (the rows of `layout.demands`) |
| `sinks.k` | 1 entry | commodity demanded |
| `sinks.backlog` | 1 entry | true: unserved demand is carried to later weeks; false: it is lost |
| `sinks.pi` | 1 entry | shortage penalty pi, USD per unit of demand not served |
| `dyads.a` | 1 entry | first region index of each dyad (the warning's dyad units) |
| `dyads.b` | 1 entry | second region index of each dyad |
| `regime` | `standard` | the information regime's published parameters (`name`, `L`, `a`, `phi`, `chi`, `h_cov`, `skill`, `blackout`); null when the runner hides it |

--- 