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

Generated for instance `chokepoint-small` (T = 52, regime `standard`) by `uv run python scripts/fields_docs.py`.

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

**Action slots** (`flows`, `action_mask`, `slot_mask`, `last_week.clip.*`):

| slot | edge | from | to | commodity | lane |
| --- | --- | --- | --- | --- | --- |
| 0 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_tw |
| 1 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_kr |
| 2 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_eu |
| 3 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_eu.cape |
| 4 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_tw.lombok |
| 5 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_kr.lombok |
| 6 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_kr.east |
| 7 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_kr |
| 8 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_jp |
| 9 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_kr.lombok |
| 10 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_jp.lombok |
| 11 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_kr.east |
| 12 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_jp.east |
| 13 | bypass.tb.src_gulf_crude.chk_malacca | src_gulf_crude | chk_malacca | crude | lane.src_gulf_crude.term_kr.bypass |
| 14 | bypass.tb.src_gulf_crude.chk_malacca | src_gulf_crude | chk_malacca | crude | lane.src_gulf_crude.term_jp.bypass |
| 15 | sea.tb.src_us_lng.chk_panama | src_us_lng | chk_panama | lng | lane.src_us_lng.term_tw |
| 16 | sea.tb.src_us_lng.chk_panama | src_us_lng | chk_panama | lng | lane.src_us_lng.term_jp |
| 17 | sea.tb.src_us_lng.term_eu | src_us_lng | term_eu | lng | - |
| 18 | sea.tb.src_us_crude.chk_panama | src_us_crude | chk_panama | crude | lane.src_us_crude.term_tw |
| 19 | sea.tb.src_us_crude.term_eu | src_us_crude | term_eu | crude | - |
| 20 | sea.tb.src_au_lng.term_kr | src_au_lng | term_kr | lng | - |
| 21 | sea.tb.src_au_lng.term_jp | src_au_lng | term_jp | lng | - |
| 22 | sea.tb.src_ru_gas.term_jp | src_ru_gas | term_jp | lng | - |
| 23 | pipe.tb.src_ru_gas.grid_eu | src_ru_gas | grid_eu | lng | - |
| 24 | pipe.tb.src_kz_uranium.grid_kr | src_kz_uranium | grid_kr | nucfuel | - |
| 25 | pipe.tb.src_kz_uranium.grid_jp | src_kz_uranium | grid_jp | nucfuel | - |
| 26 | pipe.tb.src_kz_uranium.grid_eu | src_kz_uranium | grid_eu | nucfuel | - |
| 27 | tg.tb.term_tw.grid_tw | term_tw | grid_tw | lng | - |
| 28 | tg.tb.term_tw.grid_tw | term_tw | grid_tw | crude | - |
| 29 | tg.tb.term_kr.grid_kr | term_kr | grid_kr | lng | - |
| 30 | tg.tb.term_kr.grid_kr | term_kr | grid_kr | crude | - |
| 31 | tg.tb.term_jp.grid_jp | term_jp | grid_jp | lng | - |
| 32 | tg.tb.term_jp.grid_jp | term_jp | grid_jp | crude | - |
| 33 | tg.tb.term_eu.grid_eu | term_eu | grid_eu | lng | - |
| 34 | tg.tb.term_eu.grid_eu | term_eu | grid_eu | crude | - |
| 35 | sea.ct.mat_jp_wafer.fab_tw_leading_1 | mat_jp_wafer | fab_tw_leading_1 | wafer | - |
| 36 | air.ct.mat_jp_wafer.fab_tw_leading_1 | mat_jp_wafer | fab_tw_leading_1 | wafer | - |
| 37 | sea.ct.mat_jp_wafer.fab_tw_mature_1 | mat_jp_wafer | fab_tw_mature_1 | wafer | - |
| 38 | air.ct.mat_jp_wafer.fab_tw_mature_1 | mat_jp_wafer | fab_tw_mature_1 | wafer | - |
| 39 | sea.ct.mat_jp_wafer.fab_kr_memory_1 | mat_jp_wafer | fab_kr_memory_1 | wafer | - |
| 40 | air.ct.mat_jp_wafer.fab_kr_memory_1 | mat_jp_wafer | fab_kr_memory_1 | wafer | - |
| 41 | sea.ct.mat_jp_wafer.fab_jp_memory_1 | mat_jp_wafer | fab_jp_memory_1 | wafer | - |
| 42 | air.ct.mat_jp_wafer.fab_jp_memory_1 | mat_jp_wafer | fab_jp_memory_1 | wafer | - |
| 43 | sea.ct.mat_de_wafer.fab_eu_leading_1 | mat_de_wafer | fab_eu_leading_1 | wafer | - |
| 44 | air.ct.mat_de_wafer.fab_eu_leading_1 | mat_de_wafer | fab_eu_leading_1 | wafer | - |
| 45 | sea.ct.mat_de_wafer.fab_eu_mature_1 | mat_de_wafer | fab_eu_mature_1 | wafer | - |
| 46 | air.ct.mat_de_wafer.fab_eu_mature_1 | mat_de_wafer | fab_eu_mature_1 | wafer | - |
| 47 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_tw_leading_1 |
| 48 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1 |
| 49 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_eu_leading_1 |
| 50 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_eu_mature_1 |
| 51 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_tw_leading_1.cape |
| 52 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1.cape |
| 53 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_tw_leading_1.lombok |
| 54 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1.lombok |
| 55 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1.east |
| 56 | air.ct.mat_ua_neon.fab_tw_leading_1 | mat_ua_neon | fab_tw_leading_1 | wafer | - |
| 57 | air.ct.mat_ua_neon.fab_kr_memory_1 | mat_ua_neon | fab_kr_memory_1 | wafer | - |
| 58 | air.ct.mat_ua_neon.fab_eu_leading_1 | mat_ua_neon | fab_eu_leading_1 | wafer | - |
| 59 | air.ct.mat_ua_neon.fab_eu_mature_1 | mat_ua_neon | fab_eu_mature_1 | wafer | - |
| 60 | sea.ct.fab_tw_leading_1.osat_my | fab_tw_leading_1 | osat_my | chip_le_raw | - |
| 61 | air.ct.fab_tw_leading_1.osat_my | fab_tw_leading_1 | osat_my | chip_le_raw | - |
| 62 | sea.ct.fab_tw_leading_1.osat_tw | fab_tw_leading_1 | osat_tw | chip_le_raw | - |
| 63 | air.ct.fab_tw_leading_1.osat_tw | fab_tw_leading_1 | osat_tw | chip_le_raw | - |
| 64 | sea.ct.fab_tw_mature_1.osat_tw | fab_tw_mature_1 | osat_tw | chip_mat_raw | - |
| 65 | air.ct.fab_tw_mature_1.osat_tw | fab_tw_mature_1 | osat_tw | chip_mat_raw | - |
| 66 | sea.ct.fab_kr_memory_1.chk_taiwan | fab_kr_memory_1 | chk_taiwan | chip_le_raw | lane.fab_kr_memory_1.osat_my |
| 67 | east.ct.fab_kr_memory_1.osat_my | fab_kr_memory_1 | osat_my | chip_le_raw | - |
| 68 | air.ct.fab_kr_memory_1.osat_my | fab_kr_memory_1 | osat_my | chip_le_raw | - |
| 69 | sea.ct.fab_kr_memory_1.osat_kr | fab_kr_memory_1 | osat_kr | chip_le_raw | - |
| 70 | air.ct.fab_kr_memory_1.osat_kr | fab_kr_memory_1 | osat_kr | chip_le_raw | - |
| 71 | sea.ct.fab_jp_memory_1.chk_taiwan | fab_jp_memory_1 | chk_taiwan | chip_le_raw | lane.fab_jp_memory_1.osat_my |
| 72 | east.ct.fab_jp_memory_1.osat_my | fab_jp_memory_1 | osat_my | chip_le_raw | - |
| 73 | air.ct.fab_jp_memory_1.osat_my | fab_jp_memory_1 | osat_my | chip_le_raw | - |
| 74 | sea.ct.fab_eu_leading_1.chk_suez | fab_eu_leading_1 | chk_suez | chip_le_raw | lane.fab_eu_leading_1.osat_my |
| 75 | sea.ct.fab_eu_leading_1.chk_suez | fab_eu_leading_1 | chk_suez | chip_le_raw | lane.fab_eu_leading_1.osat_my.lombok |
| 76 | cape.ct.fab_eu_leading_1.chk_cape | fab_eu_leading_1 | chk_cape | chip_le_raw | lane.fab_eu_leading_1.osat_my.cape |
| 77 | air.ct.fab_eu_leading_1.osat_my | fab_eu_leading_1 | osat_my | chip_le_raw | - |
| 78 | sea.ct.fab_eu_mature_1.chk_suez | fab_eu_mature_1 | chk_suez | chip_mat_raw | lane.fab_eu_mature_1.osat_my |
| 79 | sea.ct.fab_eu_mature_1.chk_suez | fab_eu_mature_1 | chk_suez | chip_mat_raw | lane.fab_eu_mature_1.osat_my.lombok |
| 80 | cape.ct.fab_eu_mature_1.chk_cape | fab_eu_mature_1 | chk_cape | chip_mat_raw | lane.fab_eu_mature_1.osat_my.cape |
| 81 | air.ct.fab_eu_mature_1.osat_my | fab_eu_mature_1 | osat_my | chip_mat_raw | - |
| 82 | sea.ct.osat_my.chk_malacca | osat_my | chk_malacca | chip_mat | lane.osat_my.sink_eu |
| 83 | sea.ct.osat_my.chk_taiwan | osat_my | chk_taiwan | chip_mat | lane.osat_my.sink_cn |
| 84 | sea.ct.osat_my.chk_taiwan | osat_my | chk_taiwan | chip_mat | lane.osat_my.sink_jp |
| 85 | sea.ct.osat_my.sink_us | osat_my | sink_us | chip_mat | - |
| 86 | air.ct.osat_my.sink_us | osat_my | sink_us | chip_le | - |
| 87 | air.ct.osat_my.sink_us | osat_my | sink_us | chip_mat | - |
| 88 | air.ct.osat_my.sink_eu | osat_my | sink_eu | chip_le | - |
| 89 | air.ct.osat_my.sink_eu | osat_my | sink_eu | chip_mat | - |
| 90 | air.ct.osat_my.sink_cn | osat_my | sink_cn | chip_le | - |
| 91 | air.ct.osat_my.sink_jp | osat_my | sink_jp | chip_le | - |
| 92 | air.ct.osat_my.sink_jp | osat_my | sink_jp | chip_mat | - |
| 93 | sea.ct.osat_tw.chk_malacca | osat_tw | chk_malacca | chip_mat | lane.osat_tw.sink_eu |
| 94 | sea.ct.osat_tw.sink_us | osat_tw | sink_us | chip_mat | - |
| 95 | air.ct.osat_tw.sink_us | osat_tw | sink_us | chip_le | - |
| 96 | air.ct.osat_tw.sink_us | osat_tw | sink_us | chip_mat | - |
| 97 | air.ct.osat_tw.sink_eu | osat_tw | sink_eu | chip_le | - |
| 98 | air.ct.osat_tw.sink_eu | osat_tw | sink_eu | chip_mat | - |
| 99 | sea.ct.osat_tw.sink_cn | osat_tw | sink_cn | chip_mat | - |
| 100 | air.ct.osat_tw.sink_cn | osat_tw | sink_cn | chip_le | - |
| 101 | sea.ct.osat_tw.sink_jp | osat_tw | sink_jp | chip_mat | - |
| 102 | air.ct.osat_tw.sink_jp | osat_tw | sink_jp | chip_le | - |
| 103 | air.ct.osat_tw.sink_jp | osat_tw | sink_jp | chip_mat | - |
| 104 | air.ct.osat_kr.sink_us | osat_kr | sink_us | chip_le | - |
| 105 | air.ct.osat_kr.sink_eu | osat_kr | sink_eu | chip_le | - |
| 106 | air.ct.osat_kr.sink_cn | osat_kr | sink_cn | chip_le | - |
| 107 | air.ct.osat_kr.sink_jp | osat_kr | sink_jp | chip_le | - |

**Override slots** (`override_qty`, `override_mask`):

| slot | chokepoint | commodity | out edge | lane |
| --- | --- | --- | --- | --- |
| 0 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_malacca | lane.src_qa_lng.term_tw |
| 1 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_malacca | lane.src_qa_lng.term_kr |
| 2 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_malacca | lane.src_qa_lng.term_kr.east |
| 3 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_suez | lane.src_qa_lng.term_eu |
| 4 | chk_hormuz | lng | cape.tb.chk_hormuz.chk_cape | lane.src_qa_lng.term_eu.cape |
| 5 | chk_hormuz | lng | lombok.tb.chk_hormuz.chk_taiwan | lane.src_qa_lng.term_kr.lombok |
| 6 | chk_hormuz | lng | lombok.tb.chk_hormuz.term_tw | lane.src_qa_lng.term_tw.lombok |
| 7 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_kr |
| 8 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_jp |
| 9 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_kr.east |
| 10 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_jp.east |
| 11 | chk_hormuz | crude | lombok.tb.chk_hormuz.chk_taiwan | lane.src_gulf_crude.term_kr.lombok |
| 12 | chk_hormuz | crude | lombok.tb.chk_hormuz.chk_taiwan | lane.src_gulf_crude.term_jp.lombok |
| 13 | chk_malacca | lng | sea.tb.chk_malacca.chk_taiwan | lane.src_qa_lng.term_kr |
| 14 | chk_malacca | lng | sea.tb.chk_malacca.term_tw | lane.src_qa_lng.term_tw |
| 15 | chk_malacca | lng | east.tb.chk_malacca.term_kr | lane.src_qa_lng.term_kr.east |
| 16 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_kr |
| 17 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_jp |
| 18 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_kr.bypass |
| 19 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_jp.bypass |
| 20 | chk_malacca | crude | east.tb.chk_malacca.term_kr | lane.src_gulf_crude.term_kr.east |
| 21 | chk_malacca | crude | east.tb.chk_malacca.term_jp | lane.src_gulf_crude.term_jp.east |
| 22 | chk_suez | lng | sea.tb.chk_suez.term_eu | lane.src_qa_lng.term_eu |
| 23 | chk_suez | lng | turnback.tb.chk_suez.term_eu | - |
| 24 | chk_cape | lng | cape.tb.chk_cape.term_eu | lane.src_qa_lng.term_eu.cape |
| 25 | chk_taiwan | lng | sea.tb.chk_taiwan.term_kr | lane.src_qa_lng.term_kr |
| 26 | chk_taiwan | lng | sea.tb.chk_taiwan.term_kr | lane.src_qa_lng.term_kr.lombok |
| 27 | chk_taiwan | crude | sea.tb.chk_taiwan.term_kr | lane.src_gulf_crude.term_kr |
| 28 | chk_taiwan | crude | sea.tb.chk_taiwan.term_kr | lane.src_gulf_crude.term_kr.lombok |
| 29 | chk_taiwan | crude | sea.tb.chk_taiwan.term_kr | lane.src_gulf_crude.term_kr.bypass |
| 30 | chk_taiwan | crude | sea.tb.chk_taiwan.term_jp | lane.src_gulf_crude.term_jp |
| 31 | chk_taiwan | crude | sea.tb.chk_taiwan.term_jp | lane.src_gulf_crude.term_jp.lombok |
| 32 | chk_taiwan | crude | sea.tb.chk_taiwan.term_jp | lane.src_gulf_crude.term_jp.bypass |
| 33 | chk_panama | lng | sea.tb.chk_panama.term_tw | lane.src_us_lng.term_tw |
| 34 | chk_panama | lng | sea.tb.chk_panama.term_jp | lane.src_us_lng.term_jp |
| 35 | chk_panama | crude | sea.tb.chk_panama.term_tw | lane.src_us_crude.term_tw |

**Layout tables** (`config['layout']`, the positions of the densified blocks):

| table | entries |
| --- | --- |
| `stock_slots` | 0: src_qa_lng/lng, 1: src_gulf_crude/crude, 2: src_us_lng/lng, 3: src_us_crude/crude, 4: src_au_lng/lng, 5: src_ru_gas/lng, 6: src_kz_uranium/nucfuel, 7: term_tw/lng, 8: term_tw/crude, 9: term_kr/lng, 10: term_kr/crude, 11: term_jp/lng, 12: term_jp/crude, 13: term_eu/lng, 14: term_eu/crude, 15: grid_tw/lng, 16: grid_tw/crude, 17: grid_kr/lng, 18: grid_kr/crude, 19: grid_kr/nucfuel, 20: grid_jp/lng, 21: grid_jp/crude, 22: grid_jp/nucfuel, 23: grid_eu/lng, 24: grid_eu/crude, 25: grid_eu/nucfuel, 26: mat_jp_wafer/wafer, 27: mat_de_wafer/wafer, 28: mat_ua_neon/wafer, 29: fab_tw_leading_1/wafer, 30: fab_tw_leading_1/chip_le_raw, 31: fab_tw_mature_1/wafer, 32: fab_tw_mature_1/chip_mat_raw, 33: fab_kr_memory_1/wafer, 34: fab_kr_memory_1/chip_le_raw, 35: fab_jp_memory_1/wafer, 36: fab_jp_memory_1/chip_le_raw, 37: fab_eu_leading_1/wafer, 38: fab_eu_leading_1/chip_le_raw, 39: fab_eu_mature_1/wafer, 40: fab_eu_mature_1/chip_mat_raw, 41: osat_my/chip_le_raw, 42: osat_my/chip_mat_raw, 43: osat_my/chip_le, 44: osat_my/chip_mat, 45: osat_tw/chip_le_raw, 46: osat_tw/chip_mat_raw, 47: osat_tw/chip_le, 48: osat_tw/chip_mat, 49: osat_kr/chip_le_raw, 50: osat_kr/chip_le, 51: sink_us/chip_le, 52: sink_us/chip_mat, 53: sink_eu/chip_le, 54: sink_eu/chip_mat, 55: sink_cn/chip_le, 56: sink_cn/chip_mat, 57: sink_jp/chip_le, 58: sink_jp/chip_mat |
| `supply_slots` | 0: src_qa_lng/lng, 1: src_gulf_crude/crude, 2: src_us_lng/lng, 3: src_us_crude/crude, 4: src_au_lng/lng, 5: src_ru_gas/lng, 6: src_kz_uranium/nucfuel, 7: mat_jp_wafer/wafer, 8: mat_de_wafer/wafer, 9: mat_ua_neon/wafer |
| `demands` | 0: sink_us/chip_le, 1: sink_us/chip_mat, 2: sink_eu/chip_le, 3: sink_eu/chip_mat, 4: sink_cn/chip_le, 5: sink_cn/chip_mat, 6: sink_jp/chip_le, 7: sink_jp/chip_mat |
| `chokepoints` | 0: chk_hormuz, 1: chk_malacca, 2: chk_suez, 3: chk_cape, 4: chk_taiwan, 5: chk_panama, 6: chk_turkish |
| `fabs` | 0: fab_tw_leading_1, 1: fab_tw_mature_1, 2: fab_kr_memory_1, 3: fab_jp_memory_1, 4: fab_eu_leading_1, 5: fab_eu_mature_1 |
| `grids` | 0: grid_tw, 1: grid_kr, 2: grid_jp, 3: grid_eu |
| `osats` | 0: osat_my, 1: osat_tw, 2: osat_kr |
| `warning_units` | 0: region TW, 1: region KR, 2: region JP, 3: region CN, 4: region US, 5: region EU, 6: region GULF, 7: region RU, 8: region AU, 9: region SEA, 10: region IN, 11: region UA, 12: region KZ, 13: region ROW, 14: dyad 0, 15: chokepoint chk_hormuz, 16: chokepoint chk_malacca, 17: chokepoint chk_suez, 18: chokepoint chk_cape, 19: chokepoint chk_taiwan, 20: chokepoint chk_panama, 21: chokepoint chk_turkish |
| `cost_components` | 0: freight, 1: war_risk, 2: tariff, 3: holding, 4: queue_holding, 5: shortage, 6: disposal, 7: shed |
| `release_pairs` | 0: chk_hormuz/lng, 1: chk_hormuz/crude, 2: chk_malacca/lng, 3: chk_malacca/crude, 4: chk_suez/lng, 5: chk_suez/crude, 6: chk_cape/lng, 7: chk_cape/crude, 8: chk_taiwan/lng, 9: chk_taiwan/crude, 10: chk_panama/lng, 11: chk_panama/crude, 12: chk_turkish/lng, 13: chk_turkish/crude |
| `lot_keys` | 0: chk_hormuz/lng/lane.src_qa_lng.term_tw/sea.tb.chk_hormuz.chk_malacca, 1: chk_hormuz/lng/lane.src_qa_lng.term_kr/sea.tb.chk_hormuz.chk_malacca, 2: chk_hormuz/lng/lane.src_qa_lng.term_eu/sea.tb.chk_hormuz.chk_suez, 3: chk_hormuz/lng/lane.src_qa_lng.term_eu.cape/cape.tb.chk_hormuz.chk_cape, 4: chk_hormuz/lng/lane.src_qa_lng.term_tw.lombok/lombok.tb.chk_hormuz.term_tw, 5: chk_hormuz/lng/lane.src_qa_lng.term_kr.lombok/lombok.tb.chk_hormuz.chk_taiwan, 6: chk_hormuz/lng/lane.src_qa_lng.term_kr.east/sea.tb.chk_hormuz.chk_malacca, 7: chk_hormuz/crude/lane.src_gulf_crude.term_kr/sea.tb.chk_hormuz.chk_malacca, 8: chk_hormuz/crude/lane.src_gulf_crude.term_jp/sea.tb.chk_hormuz.chk_malacca, 9: chk_hormuz/crude/lane.src_gulf_crude.term_kr.lombok/lombok.tb.chk_hormuz.chk_taiwan, 10: chk_hormuz/crude/lane.src_gulf_crude.term_jp.lombok/lombok.tb.chk_hormuz.chk_taiwan, 11: chk_hormuz/crude/lane.src_gulf_crude.term_kr.east/sea.tb.chk_hormuz.chk_malacca, 12: chk_hormuz/crude/lane.src_gulf_crude.term_jp.east/sea.tb.chk_hormuz.chk_malacca, 13: chk_malacca/lng/lane.src_qa_lng.term_tw/sea.tb.chk_malacca.term_tw, 14: chk_malacca/lng/lane.src_qa_lng.term_kr/sea.tb.chk_malacca.chk_taiwan, 15: chk_malacca/lng/lane.src_qa_lng.term_kr.east/east.tb.chk_malacca.term_kr, 16: chk_malacca/crude/lane.src_gulf_crude.term_kr/sea.tb.chk_malacca.chk_taiwan, 17: chk_malacca/crude/lane.src_gulf_crude.term_jp/sea.tb.chk_malacca.chk_taiwan, 18: chk_malacca/crude/lane.src_gulf_crude.term_kr.east/east.tb.chk_malacca.term_kr, 19: chk_malacca/crude/lane.src_gulf_crude.term_jp.east/east.tb.chk_malacca.term_jp, 20: chk_malacca/crude/lane.src_gulf_crude.term_kr.bypass/sea.tb.chk_malacca.chk_taiwan, 21: chk_malacca/crude/lane.src_gulf_crude.term_jp.bypass/sea.tb.chk_malacca.chk_taiwan, 22: chk_malacca/wafer/lane.mat_ua_neon.fab_tw_leading_1/sea.ct.chk_malacca.fab_tw_leading_1, 23: chk_malacca/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_malacca.chk_taiwan, 24: chk_malacca/wafer/lane.mat_ua_neon.fab_tw_leading_1.cape/sea.ct.chk_malacca.fab_tw_leading_1, 25: chk_malacca/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/sea.ct.chk_malacca.chk_taiwan, 26: chk_malacca/wafer/lane.mat_ua_neon.fab_kr_memory_1.east/east.ct.chk_malacca.fab_kr_memory_1, 27: chk_malacca/chip_le_raw/lane.fab_eu_leading_1.osat_my/sea.ct.chk_malacca.osat_my, 28: chk_malacca/chip_le_raw/lane.fab_eu_leading_1.osat_my.cape/sea.ct.chk_malacca.osat_my, 29: chk_malacca/chip_mat_raw/lane.fab_eu_mature_1.osat_my/sea.ct.chk_malacca.osat_my, 30: chk_malacca/chip_mat_raw/lane.fab_eu_mature_1.osat_my.cape/sea.ct.chk_malacca.osat_my, 31: chk_malacca/chip_mat/lane.osat_my.sink_eu/sea.ct.chk_malacca.chk_suez, 32: chk_malacca/chip_mat/lane.osat_tw.sink_eu/sea.ct.chk_malacca.chk_suez, 33: chk_suez/lng/lane.src_qa_lng.term_eu/sea.tb.chk_suez.term_eu, 34: chk_suez/wafer/lane.mat_ua_neon.fab_tw_leading_1/sea.ct.chk_suez.chk_malacca, 35: chk_suez/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_suez.chk_malacca, 36: chk_suez/wafer/lane.mat_ua_neon.fab_tw_leading_1.lombok/lombok.ct.chk_suez.fab_tw_leading_1, 37: chk_suez/wafer/lane.mat_ua_neon.fab_kr_memory_1.lombok/lombok.ct.chk_suez.chk_taiwan, 38: chk_suez/wafer/lane.mat_ua_neon.fab_kr_memory_1.east/sea.ct.chk_suez.chk_malacca, 39: chk_suez/chip_le_raw/lane.fab_eu_leading_1.osat_my/sea.ct.chk_suez.chk_malacca, 40: chk_suez/chip_le_raw/lane.fab_eu_leading_1.osat_my.lombok/lombok.ct.chk_suez.osat_my, 41: chk_suez/chip_mat_raw/lane.fab_eu_mature_1.osat_my/sea.ct.chk_suez.chk_malacca, 42: chk_suez/chip_mat_raw/lane.fab_eu_mature_1.osat_my.lombok/lombok.ct.chk_suez.osat_my, 43: chk_suez/chip_mat/lane.osat_my.sink_eu/sea.ct.chk_suez.sink_eu, 44: chk_suez/chip_mat/lane.osat_tw.sink_eu/sea.ct.chk_suez.sink_eu, 45: chk_cape/lng/lane.src_qa_lng.term_eu.cape/cape.tb.chk_cape.term_eu, 46: chk_cape/wafer/lane.mat_ua_neon.fab_tw_leading_1.cape/cape.ct.chk_cape.chk_malacca, 47: chk_cape/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/cape.ct.chk_cape.chk_malacca, 48: chk_cape/chip_le_raw/lane.fab_eu_leading_1.osat_my.cape/cape.ct.chk_cape.chk_malacca, 49: chk_cape/chip_mat_raw/lane.fab_eu_mature_1.osat_my.cape/cape.ct.chk_cape.chk_malacca, 50: chk_taiwan/lng/lane.src_qa_lng.term_kr/sea.tb.chk_taiwan.term_kr, 51: chk_taiwan/lng/lane.src_qa_lng.term_kr.lombok/sea.tb.chk_taiwan.term_kr, 52: chk_taiwan/crude/lane.src_gulf_crude.term_kr/sea.tb.chk_taiwan.term_kr, 53: chk_taiwan/crude/lane.src_gulf_crude.term_jp/sea.tb.chk_taiwan.term_jp, 54: chk_taiwan/crude/lane.src_gulf_crude.term_kr.lombok/sea.tb.chk_taiwan.term_kr, 55: chk_taiwan/crude/lane.src_gulf_crude.term_jp.lombok/sea.tb.chk_taiwan.term_jp, 56: chk_taiwan/crude/lane.src_gulf_crude.term_kr.bypass/sea.tb.chk_taiwan.term_kr, 57: chk_taiwan/crude/lane.src_gulf_crude.term_jp.bypass/sea.tb.chk_taiwan.term_jp, 58: chk_taiwan/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_taiwan.fab_kr_memory_1, 59: chk_taiwan/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/sea.ct.chk_taiwan.fab_kr_memory_1, 60: chk_taiwan/wafer/lane.mat_ua_neon.fab_kr_memory_1.lombok/sea.ct.chk_taiwan.fab_kr_memory_1, 61: chk_taiwan/chip_le_raw/lane.fab_kr_memory_1.osat_my/sea.ct.chk_taiwan.osat_my, 62: chk_taiwan/chip_le_raw/lane.fab_jp_memory_1.osat_my/sea.ct.chk_taiwan.osat_my, 63: chk_taiwan/chip_mat/lane.osat_my.sink_cn/sea.ct.chk_taiwan.sink_cn, 64: chk_taiwan/chip_mat/lane.osat_my.sink_jp/sea.ct.chk_taiwan.sink_jp, 65: chk_panama/lng/lane.src_us_lng.term_tw/sea.tb.chk_panama.term_tw, 66: chk_panama/lng/lane.src_us_lng.term_jp/sea.tb.chk_panama.term_jp, 67: chk_panama/crude/lane.src_us_crude.term_tw/sea.tb.chk_panama.term_tw, 68: chk_turkish/wafer/lane.mat_ua_neon.fab_tw_leading_1/sea.ct.chk_turkish.chk_suez, 69: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_turkish.chk_suez, 70: chk_turkish/wafer/lane.mat_ua_neon.fab_eu_leading_1/sea.ct.chk_turkish.fab_eu_leading_1, 71: chk_turkish/wafer/lane.mat_ua_neon.fab_eu_mature_1/sea.ct.chk_turkish.fab_eu_mature_1, 72: chk_turkish/wafer/lane.mat_ua_neon.fab_tw_leading_1.cape/cape.ct.chk_turkish.chk_cape, 73: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/cape.ct.chk_turkish.chk_cape, 74: chk_turkish/wafer/lane.mat_ua_neon.fab_tw_leading_1.lombok/sea.ct.chk_turkish.chk_suez, 75: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1.lombok/sea.ct.chk_turkish.chk_suez, 76: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1.east/sea.ct.chk_turkish.chk_suez |

**Static tables** (`config['static']`, the episode's public tables): a table is a dict of equal-length lists, one entry per node, edge, lane, commodity, slot or sink, and the indices above point into them.

| field | here | meaning |
| --- | --- | --- |
| `instance` | keys `T`, `chokepoint_adjacency`, `commodities`, `compatibility`, `edges`, `initial_state`, `instance_id`, `kind`, `lanes`, `nodes`, `params`, `prohibitions_at_reset`, `provenance`, `region_class`, `regions`, `routing_table`, `schema_version`, `trade_adjacency`, `units`, `use` | the full public instance JSON; `initial_state.pipeline` holds the week-0 shipments of the nominal plan (the nominal flows the heuristic sample reads); edges there call the lead time `tau` |
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