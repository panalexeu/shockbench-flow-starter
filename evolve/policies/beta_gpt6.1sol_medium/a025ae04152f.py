# -0.30528217627243315
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        e = self.s['edges']
        a = self.s['action_slots']
        self.tail = np.asarray(e['tail'], dtype=int)
        self.head = np.asarray(e['head'], dtype=int)
        self.tau0 = np.asarray(e['tau0'], dtype=float)
        self.c0 = np.asarray(e['c0'], dtype=float)
        self.u0 = np.asarray([float(x) if x is not None else 0.0 for x in e['u0']])
        self.value = np.asarray(self.s['commodities']['v'], dtype=float)
        self.n = len(a['edge'])
        self.routes = []
        self.groups = {}
        self.stock_slots = [tuple(map(int, x)) for x in self.l['stock_slots']]
        self.supply_slots = [tuple(map(int, x)) for x in self.l['supply_slots']]
        self.demands = [tuple(map(int, x)) for x in self.l['demands']]
        self.cpindex = {int(n): i for i, n in enumerate(self.l['chokepoints'])}
        self.nodes_type = self.s['nodes']['type']
        for i, (edge, k, lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            edge, k = int(edge), int(k)
            lane = -1 if lane is None else int(lane)
            path = [edge] if lane < 0 else list(map(int, self.s['lanes']['edges'][lane]))
            dest = int(self.head[path[-1]])
            origin = int(self.tail[path[0]])
            cps = [] if lane < 0 else list(map(int, self.s['lanes']['chokepoints'][lane]))
            r = (origin, dest, k, lane, path, cps)
            self.routes.append(r)
            self.groups.setdefault((dest, k), []).append(i)
        self.lane_dest = {i: int(self.head[int(es[-1])]) for i, es in enumerate(self.s['lanes']['edges']) if len(es)}
        self.cache = {}
        self.rate = None
        self.initial_demand = {}
        self.safety = {}
        self.penalty = {}
        sink = self.s['sinks']
        for node, k, pi in zip(sink['node'], sink['k'], sink['pi']):
            self.penalty[(int(node), int(k))] = float(pi)

    def graph(self, obs, name, default):
        x = np.asarray(obs.get(name, default)).copy()
        mask = obs.get(name + '.observed')
        old = self.cache.get(name, np.asarray(default))
        if mask is not None and np.shape(mask) == x.shape:
            x = np.where(np.asarray(mask, dtype=bool), x, old)
        self.cache[name] = x.copy()
        return x

    def live(self, obs, key):
        x = np.asarray(obs.get(key, []))
        m = np.asarray(obs.get(key + '.observed', np.ones(x.shape)), dtype=bool)
        return np.flatnonzero(m & (x > 0))

    def initialize(self, obs):
        E = len(self.head)
        K = len(self.value)
        edge_rate = np.zeros((E, K))
        lane_edge = {}
        pq = np.asarray(obs.get('pipeline.qty', []))
        pe = np.asarray(obs.get('pipeline.edge', []), dtype=int)
        pk = np.asarray(obs.get('pipeline.k', []), dtype=int)
        pl = np.asarray(obs.get('pipeline.lane', np.full(len(pq), -1)), dtype=int)
        lm = np.asarray(obs.get('pipeline.lane.observed', np.ones(len(pq))), dtype=bool)
        for j in self.live(obs, 'pipeline.qty'):
            e, k = int(pe[j]), int(pk[j])
            if not (0 <= e < E and 0 <= k < K):
                continue
            edge_rate[e, k] += float(pq[j]) / max(1.0, self.tau0[e])
            if lm[j] and pl[j] >= 0:
                key = (int(pl[j]), e, k)
                lane_edge[key] = lane_edge.get(key, 0.0) + float(pq[j]) / max(1.0, self.tau0[e])
        base = np.zeros(self.n)
        for i, (_, _, k, lane, path, _) in enumerate(self.routes):
            if lane >= 0:
                vals = [lane_edge.get((lane, e, k), 0.0) for e in path]
                positive = [v for v in vals if v > 0]
                base[i] = float(np.median(positive)) if positive else 0.0
            else:
                base[i] = edge_rate[path[0], k]
        incoming, outgoing = {}, {}
        for i, (origin, dest, k, _, _, _) in enumerate(self.routes):
            incoming[(dest, k)] = incoming.get((dest, k), 0.0) + base[i]
            outgoing[(origin, k)] = outgoing.get((origin, k), 0.0) + base[i]
        self.rate = {}
        fq = np.asarray(obs.get('demand_forecast.qty', np.zeros((len(self.demands), 1))))
        for key, ids in self.groups.items():
            rate = max(incoming.get(key, 0.0), outgoing.get(key, 0.0))
            if rate <= 0:
                # An empty initial pipeline should not permanently disable a route.
                rate = 0.15 * max((self.u0[self.routes[i][4][0]] for i in ids), default=0.0)
            self.rate[key] = rate
        for j, key in enumerate(self.demands):
            d = float(np.mean(fq[j])) if fq.shape[1] else 0.0
            self.initial_demand[key] = max(d, 1e-9)
            self.rate[key] = d
        stock = np.asarray(obs.get('stock.qty', np.zeros(len(self.stock_slots))))
        for j, key in enumerate(self.stock_slots):
            r = self.rate.get(key, 0.0)
            self.safety[key] = min(max(float(stock[j]), 0.0), 1.25 * r)

    def act(self, observation):
        o = observation
        t = int(np.asarray(o['week']).flat[0])
        if self.rate is None:
            self.initialize(o)
        u = self.graph(o, 'graph_now.u', self.u0).astype(float)
        cost = self.graph(o, 'graph_now.c', self.c0).astype(float)
        tau = self.graph(o, 'graph_now.tau', self.tau0).astype(float)
        prohibited = self.graph(o, 'graph_now.prohibited', np.zeros((len(u), len(self.value)), dtype=np.int8))
        tariff = self.graph(o, 'graph_now.tariff', np.zeros((len(u), len(self.value))))
        opened = self.graph(o, 'graph_now.open', np.ones(len(self.cpindex)))
        mask = np.asarray(o.get('action_mask', np.ones(self.n)), dtype=bool)
        stock, available, position = {}, {}, {}
        sq = np.asarray(o.get('stock.qty', np.zeros(len(self.stock_slots))))
        for j, key in enumerate(self.stock_slots):
            q = max(0.0, float(sq[j]))
            stock[key] = q
            available[key] = q
            position[key] = q
        supply = self.graph(o, 'graph_now.supply.avail', np.zeros(len(self.supply_slots)))
        for j, key in enumerate(self.supply_slots):
            available[key] = available.get(key, 0.0) + max(0.0, float(supply[j]))
        pq = np.asarray(o.get('pipeline.qty', []))
        pe = np.asarray(o.get('pipeline.edge', []), dtype=int)
        pk = np.asarray(o.get('pipeline.k', []), dtype=int)
        pl = np.asarray(o.get('pipeline.lane', np.full(len(pq), -1)), dtype=int)
        lm = np.asarray(o.get('pipeline.lane.observed', np.ones(len(pq))), dtype=bool)
        pa = np.asarray(o.get('pipeline.arrival_week', np.full(len(pq), t + 1)), dtype=int)
        for j in self.live(o, 'pipeline.qty'):
            e, k = int(pe[j]), int(pk[j])
            if not (0 <= e < len(self.head)):
                continue
            lane = int(pl[j]) if lm[j] else -1
            dest = self.lane_dest.get(lane, int(self.head[e]))
            key = (dest, k)
            position[key] = position.get(key, 0.0) + float(pq[j])
            immediate = (int(self.head[e]), k)
            if pa[j] <= t and immediate == key:
                available[key] = available.get(key, 0.0) + float(pq[j])
        qlots = np.asarray(o.get('queue_lots.qty', []))
        for j, entry in enumerate(self.l.get('lot_keys', [])):
            if j >= len(qlots):
                break
            cp, k, lane, edge = map(int, entry)
            dest = self.lane_dest.get(lane, int(self.head[edge]))
            key = (dest, k)
            position[key] = position.get(key, 0.0) + float(np.sum(qlots[j]))
        wn = np.asarray(o.get('wip.node', []), dtype=int)
        wk = np.asarray(o.get('wip.k', []), dtype=int)
        wq = np.asarray(o.get('wip.qty', []))
        ww = np.asarray(o.get('wip.out_week', np.full(len(wq), t + 1)), dtype=int)
        for j in self.live(o, 'wip.qty'):
            key = (int(wn[j]), int(wk[j]))
            position[key] = position.get(key, 0.0) + float(wq[j])
            if ww[j] <= t:
                available[key] = available.get(key, 0.0) + float(wq[j])
        pending = {}
        pp = np.asarray(o.get('pending_prohibitions.effective_week', []))
        pm = np.asarray(o.get('pending_prohibitions.effective_week.observed', np.zeros(len(pp))), dtype=bool)
        ped = np.asarray(o.get('pending_prohibitions.edge', []), dtype=int)
        pko = np.asarray(o.get('pending_prohibitions.k', []), dtype=int)
        for j in np.flatnonzero(pm):
            key = (int(ped[j]), int(pko[j]))
            pending[key] = min(pending.get(key, self.T + 1), int(pp[j]))
        ends = {}
        ce = np.asarray(o.get('closure_end.end_week', []))
        cm = np.asarray(o.get('closure_end.end_week.observed', np.zeros(len(ce))), dtype=bool)
        cn = np.asarray(o.get('closure_end.chokepoint', []), dtype=int)
        for j in np.flatnonzero(cm):
            ends[int(cn[j])] = max(ends.get(int(cn[j]), t), int(ce[j]))
        route_info = {}
        for i, (origin, dest, k, lane, path, cps) in enumerate(self.routes):
            if not mask[i] or any(prohibited[e, k] for e in path) or u[path[0]] <= 0:
                continue
            travel = float(sum(max(0.0, tau[e]) for e in path))
            elapsed = 0.0
            unsafe = False
            for e in path:
                if pending.get((e, k), self.T + 1) <= t + elapsed:
                    unsafe = True
                elapsed += max(0.0, tau[e])
            if unsafe:
                continue
            extra = 0.0
            for cp in cps:
                openness = float(opened[self.cpindex[cp]]) if cp in self.cpindex else 1.0
                if openness < 0.95:
                    delay = max(0.0, ends.get(cp, t + 7) - t - travel * 0.5)
                    extra += delay + 2.0 * (1.0 - openness)
            lead = max(1.0, travel + extra)
            if t + lead > self.T:
                continue
            freight = float(sum(cost[e] + max(0.0, tariff[e, k]) * self.value[k] for e in path))
            time_price = max(1.0, 0.002 * self.value[k])
            score = freight + time_price * lead
            route_info[i] = (score, lead, freight)
        fq = np.asarray(o.get('demand_forecast.qty', np.zeros((len(self.demands), 1))))
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.demands))))
        demand_index = {key: j for j, key in enumerate(self.demands)}
        # Forecast changes in chip demand also propagate to their input chains.
        chip_ratios = []
        commodity_names = self.s['commodities']['id']
        for key, j in demand_index.items():
            if 'chip' in str(commodity_names[key[1]]):
                chip_ratios.append(float(np.mean(fq[j])) / self.initial_demand[key])
        chip_scale = float(np.clip(np.mean(chip_ratios), 0.6, 1.6)) if chip_ratios else 1.0
        requests = []
        for key, ids in self.groups.items():
            options = [i for i in ids if i in route_info]
            if not options:
                continue
            options.sort(key=lambda i: route_info[i][0])
            best = options[0]
            lead = route_info[best][1]
            rate = self.rate.get(key, 0.0)
            if key in demand_index:
                j = demand_index[key]
                horizon = min(fq.shape[1], max(1, int(np.ceil(lead + 1.5))), self.T - t + 1)
                forecast = fq[j, :horizon]
                rate = float(np.mean(forecast)) if len(forecast) else 0.0
                target = rate * min(lead + 1.6, self.T - t + 1) + float(backlog[j])
                priority = self.penalty.get(key, self.value[key[1]])
            else:
                name = str(commodity_names[key[1]])
                if 'chip' in name or 'wafer' in name:
                    rate *= chip_scale
                target = rate * min(lead + 1.0, max(0.0, self.T - t - 1)) + self.safety.get(key, 0.5 * rate)
                if self.T - t < lead + 3:
                    target = rate * max(0.0, self.T - t - 1)
                priority = max(10.0, self.value[key[1]])
            deficit = max(0.0, target - position.get(key, 0.0))
            if deficit > 1e-8:
                urgency = deficit / max(rate, 1.0)
                requests.append((priority * (1.0 + min(urgency, 5.0)), key, deficit, options, rate))
        requests.sort(reverse=True, key=lambda x: x[0])
        flows = np.zeros(self.n, dtype=float)
        remaining = np.maximum(u.copy(), 0.0)
        for _, key, need, options, rate in requests:
            best_lead = route_info[options[0]][1]
            for i in options:
                origin, dest, k, lane, path, cps = self.routes[i]
                source_key = (origin, k)
                inventory = available.get(source_key, 0.0)
                if self.nodes_type[origin] == 'source' and source_key not in available:
                    inventory = remaining[path[0]]
                # Longer fallback routes need an additional pipeline cushion.
                extra = min(rate, rate * max(0.0, route_info[i][1] - best_lead))
                q = min(need + extra, remaining[path[0]], inventory)
                if q <= 0:
                    continue
                flows[i] += q
                remaining[path[0]] -= q
                available[source_key] = max(0.0, inventory - q)
                need = max(0.0, need - q)
                if need <= 1e-8:
                    break
        return {'flows': np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)}
