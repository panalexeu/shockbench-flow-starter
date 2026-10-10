# 0.6844254442245871
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


class Agent:
    def __init__(self, config):
        self.c = config
        self.s = config['static']
        self.l = config['layout']
        self.raw = self.s['instance']
        self.nodes = self.raw['nodes']
        self.kid = {v: i for i, v in enumerate(self.s['commodities']['id'])}
        self.slots = [tuple(v) for v in self.l['stock_slots']]
        self.si = {v: i for i, v in enumerate(self.slots)}
        self.cp = {v: i for i, v in enumerate(self.l['chokepoints'])}
        self.fi = {v: i for i, v in enumerate(self.l['fabs'])}
        self.gi = {v: i for i, v in enumerate(self.l['grids'])}
        self.oi = {v: i for i, v in enumerate(self.l['osats'])}
        self.na = len(self.s['action_slots']['edge'])
        self.storage, self.holding, self.salvage = [], [], []
        for n, k in self.slots:
            d = self.nodes[n].get('stock', {}).get(self.s['commodities']['id'][k], {})
            self.storage.append(float(d['storage']) if d.get('storage') is not None else 1e12)
            self.holding.append(float(d.get('holding_cost', 0)))
            self.salvage.append(float(d.get('salvage', 0)))
        self.routes = []
        for j, e in enumerate(self.s['action_slots']['edge']):
            ln = self.s['action_slots']['lane'][j]
            es = list(self.s['lanes']['edges'][ln]) if ln is not None and ln >= 0 else [e]
            self.routes.append((e, self.s['action_slots']['k'][j], ln, es, self.s['edges']['tail'][e], self.s['edges']['head'][es[-1]]))
        self.cache = {}

    def act(self, o):
        def get(key, default):
            a = np.asarray(o.get(key, default), dtype=float)
            m = o.get(key + '.observed')
            if m is not None and np.shape(m) == a.shape:
                a = np.where(m, a, self.cache.get(key, np.asarray(default, dtype=float)))
            self.cache[key] = a.copy()
            return a
        week = int(o['week'][0])
        remaining = int(self.c['T']) - week + 1
        H = min(24, remaining)
        S = len(self.slots)
        E = len(self.s['edges']['id'])
        K = len(self.kid)
        stock = get('stock.qty', np.zeros(S))
        tau = np.maximum(1, get('graph_now.tau', self.s['edges']['tau0'])).astype(int)
        freight = get('graph_now.c', self.s['edges']['c0'])
        caps = get('graph_now.u', [0 if v is None else v for v in self.s['edges']['u0']])
        prohibited = get('graph_now.prohibited', np.zeros((E, K)))
        tariff = get('graph_now.tariff', np.zeros((E, K)))
        opening = get('graph_now.open', np.ones(len(self.cp)))
        war = get('graph_now.war_risk', np.zeros(len(self.cp))).astype(int)
        kap = {p: get('graph_now.kappa.' + p, np.full(len(self.cp), 1e12)) for p in ('tb', 'ct')}
        ends = {}
        for z, n in enumerate(o.get('closure_end.chokepoint', [])):
            m = o.get('closure_end.chokepoint.observed', [])
            em = o.get('closure_end.end_week.observed', [])
            if z < len(m) and m[z] and z < len(em) and em[z]:
                ends[int(n)] = max(ends.get(int(n), week), int(o['closure_end.end_week'][z]))
        def recovery(n):
            return max(0, ends.get(n, week + 4) - week + 1)
        def capacity(n, pool, h):
            ci = self.cp[n]
            value = kap[pool][ci]
            if opening[ci] < .1 and h >= recovery(n):
                a = self.nodes[n].get('chokepoint', {})
                mu = a.get('mu', {})
                if isinstance(mu, dict): value = float(a.get('k_c', 1)) * float(mu.get(pool, value))
            return max(0, value)
        pending = {}
        for z, e in enumerate(o.get('pending_prohibitions.edge', [])):
            m = o.get('pending_prohibitions.edge.observed', [])
            if z < len(m) and m[z]:
                key = (int(e), int(o['pending_prohibitions.k'][z]))
                pending[key] = min(pending.get(key, 10**9), int(o['pending_prohibitions.effective_week'][z]))
        def blocked(e, k, h):
            return prohibited[e, k] or week + h >= pending.get((e, k), 10**9)
        def suffix(e, ln):
            if ln is not None and ln >= 0:
                es = self.s['lanes']['edges'][int(ln)]
                if e in es: return list(es[es.index(e) + 1:])
            return []
        def path(es, start):
            elapsed = start
            visits = []
            for e in es:
                n = self.s['edges']['tail'][e]
                if n in self.cp and opening[self.cp[n]] < .1:
                    elapsed = max(elapsed, recovery(n))
                visits.append((e, elapsed))
                elapsed += int(tau[e])
            return elapsed, visits
        arrivals = np.zeros((S, H))
        def arrive(n, k, q, h):
            s = self.si.get((int(n), int(k)))
            if s is not None and 0 <= h < H: arrivals[s, int(h)] += float(q)
        scheduled = [[] for _ in range(H)]
        queues = {n: [] for n in self.cp}
        pq = o.get('pipeline.qty', [])
        for z in np.flatnonzero(o.get('pipeline.qty.observed', np.asarray(pq) > 0)):
            e = int(o['pipeline.edge'][z]); k = int(o['pipeline.k'][z])
            lm = o.get('pipeline.lane.observed', np.zeros(len(pq)))
            ln = int(o['pipeline.lane'][z]) if lm[z] else None
            h = int(o['pipeline.arrival_week'][z]) - week
            n = self.s['edges']['head'][e]
            rest = suffix(e, ln)
            if n in self.cp and rest:
                if 0 <= h < H: scheduled[h].append((n, k, float(pq[z]), rest, h))
            else: arrive(n, k, pq[z], h)
        for z, (n, k, ln, e) in enumerate(self.l.get('lot_keys', [])):
            if 'queue_lots.qty' not in o: continue
            for cohort in np.flatnonzero(o['queue_lots.qty'][z] > 0):
                queues[n].append([k, float(o['queue_lots.qty'][z, cohort]), [e] + suffix(e, ln), int(cohort) + 1 - week])
        used_e, used_p = {}, {}
        for h in range(H):
            for n, k, q, rest, age in scheduled[h]: queues[n].append([k, q, rest, age])
            ec = np.maximum(0, caps.copy())
            for n in self.cp:
                pc = {p: capacity(n, p, h) for p in kap}
                book = sorted(queues[n], key=lambda v: v[3])
                for lot in book:
                    k, q, rest, age = lot
                    e = rest[0]; pool = self.s['commodities']['pool'][k]
                    if blocked(e, k, h): continue
                    sent = min(q, ec[e], pc[pool])
                    if sent <= 0: continue
                    lot[1] -= sent; ec[e] -= sent; pc[pool] -= sent
                    used_e[e, h] = used_e.get((e, h), 0) + sent
                    used_p[n, pool, h] = used_p.get((n, pool, h), 0) + sent
                    hh = h + int(tau[e]); dest = self.s['edges']['head'][e]
                    if dest in self.cp and len(rest) > 1:
                        if hh < H: scheduled[hh].append((dest, k, sent, rest[1:], hh))
                    else: arrive(dest, k, sent, hh)
                queues[n] = [v for v in book if v[1] > 1e-9]
        for z, q in enumerate(o.get('wip.qty', [])):
            if q > 0 and o.get('wip.qty.observed', np.ones(len(o['wip.qty'])))[z]:
                arrive(o['wip.node'][z], o['wip.k'][z], q, int(o['wip.out_week'][z]) - week)
        obj, bounds, eq, er, ub, br = [], [], [], [], [], []
        def var(c=0, hi=None):
            j = len(obj); obj.append(float(c) / 100000); bounds.append((0, hi)); return j
        def row(rows, rhs, terms, b):
            rows.append(terms); rhs.append(float(b))
        I = [[var(self.holding[s] - (self.salvage[s] if remaining == H and h == H - 1 else 0), self.storage[s]) for h in range(H)] for s in range(S)]
        bal = [[{I[s][h]: 1} for h in range(H)] for s in range(S)]
        draws = [[{} for h in range(H)] for s in range(S)]
        def term(s, h, x, coefficient):
            if s is not None and 0 <= h < H: bal[s][h][x] = bal[s][h].get(x, 0) + coefficient
        def buffer(s, h, target, penalty):
            d = var(penalty)
            row(ub, br, {I[s][h]: -1, d: -1}, -max(0, min(target, self.storage[s] * .9)))
        for s, (n, k) in enumerate(self.slots):
            for h in range(H): term(s, h, var(float(self.raw['commodities'][k].get('disposal_cost', 3000))), 1)
        X, edge_use, pool_use = [], {}, {}
        mask = o.get('action_mask', np.ones(self.na))
        for j, (e, k, ln, es, src, dst) in enumerate(self.routes):
            a = self.si.get((src, k)); b = self.si.get((dst, k))
            price = sum(freight[x] + tariff[x, k] * self.s['commodities']['v'][k] for x in es)
            for x in es:
                n = self.s['edges']['tail'][x]
                if n in self.cp:
                    w = min(2, max(0, int(war[self.cp[n]])))
                    costs = self.nodes[n].get('chokepoint', {}).get('war_risk_cost', {}).get(self.s['commodities']['id'][k], {})
                    if isinstance(costs, dict): price += float(costs.get(('none', 'red_sea', 'hormuz_2026')[w], 0))
                    elif isinstance(costs, (list, tuple)) and w < len(costs): price += float(costs[w])
            pool = self.s['commodities']['pool'][k]; xs = []
            for h in range(H):
                arrival, visits = path(es, h)
                hi = max(0, caps[e])
                cost = price + .01 * (arrival - h)
                if arrival >= H:
                    if remaining == H and len(es) == 1 and b is not None:
                        cost -= self.salvage[b]
                    else: hi = 0
                if any(blocked(x, k, t) for x, t in visits) or a is None or b is None or (h == 0 and not mask[j]): hi = 0
                x = var(cost, hi); xs.append(x)
                term(a, h, x, 1); term(b, arrival, x, -1)
                if a is not None: draws[a][h][x] = 1
                for edge, elapsed in visits:
                    if elapsed < H:
                        edge_use.setdefault((edge, elapsed), {})[x] = 1
                        n = self.s['edges']['tail'][edge]
                        if n in self.cp: pool_use.setdefault((n, pool, elapsed), {})[x] = 1
            X.append(xs)
        for (e, h), terms in edge_use.items(): row(ub, br, terms, max(0, caps[e] - used_e.get((e, h), 0)))
        for (n, pool, h), terms in pool_use.items(): row(ub, br, terms, max(0, capacity(n, pool, h) - used_p.get((n, pool, h), 0)))
        supply = get('graph_now.supply.avail', np.zeros(len(self.l['supply_slots'])))
        for z, key in enumerate(self.l['supply_slots']):
            s = self.si.get(tuple(key))
            if s is not None:
                for h in range(H): term(s, h, var(0, max(0, supply[z])), -1)
        R = get('graph_now.fab.R', np.ones(len(self.fi)))
        alpha = get('graph_now.fab.alpha_bar', np.ones(len(self.fi)))
        G = get('graph_now.grid.G_bar', [self.nodes[n]['grid']['deliverable'] for n in self.gi])
        Y = get('graph_now.grid.y_bar', [self.nodes[n]['grid']['base_load'] for n in self.gi])
        targets, fuel_rates = {}, {}
        for n, gi in self.gi.items():
            g = self.nodes[n]['grid']; load = float(Y[gi])
            for fn, fi in self.fi.items():
                f = self.nodes[fn]['fab']
                if f.get('grid') == self.s['nodes']['id'][n]: load += float(f['e']) * float(f['cap0']) * float(alpha[fi]) if R[fi] > 0 else 0
            fraction = min(1, load / max(1e-12, G[gi]))
            for fuel, share in g['shares'].items():
                if fuel not in self.kid: continue
                s = self.si.get((n, self.kid[fuel]))
                if s is None: continue
                rate = float(share) * float(G[gi]) * fraction
                fuel_rates[s] = rate
                ibar = float(g.get('ibar', {}).get(fuel, 0)); targets[s] = max(ibar, rate * 4)
                threshold = float(self.raw['params'].get('psi', 0)) * ibar
                for h in range(H):
                    burn = var(0, rate); lack = var(float(g.get('voll', 4100000)), rate)
                    term(s, h, burn, 1); row(eq, er, {burn: 1, lack: 1}, rate)
                    if fuel == g.get('rationed') and threshold > 0:
                        mx = float(share) * float(G[gi]) / threshold
                        if h == 0: row(ub, br, {burn: 1}, max(0, mx * stock[s]))
                        else: row(ub, br, {burn: 1, I[s][h - 1]: -mx}, 0)
                    if remaining - h > 1:
                        target = max(threshold * 1.15 if fuel == g.get('rationed') else 0, rate * 2.7)
                        buffer(s, h, min(target, rate * (remaining - h - 1)), 14000)
        for sn, k in self.slots:
            if self.nodes[sn]['type'] != 'terminal': continue
            ss = self.si[sn, k]
            rates = []
            for e, kk, ln, es, src, dst in self.routes:
                if src == sn and kk == k and (dst, k) in self.si:
                    ds = self.si[dst, k]
                    if ds in fuel_rates: rates.append(fuel_rates[ds])
            if rates:
                rate = sum(rates)
                for h in range(H):
                    if remaining - h > 3: buffer(ss, h, min(rate * 1.2, rate * (remaining - h - 3)), 2200)
        for n, fi in self.fi.items():
            f = self.nodes[n]['fab']; a = self.si[n, self.kid[f['input']]]; b = self.si[n, self.kid[f['product']]]
            cap = max(0, float(f['cap0']) * R[fi] * alpha[fi]); targets[a] = min(self.storage[a] * .8, float(f['cap0']) * 2)
            for h in range(H):
                late = h + int(f['tau']) >= H
                credit = self.salvage[a] if late and remaining == H else 0
                x = var(-credit, cap); term(a, h, x, 1); term(b, h + int(f['tau']), x, -1)
                if remaining - h > int(f['tau']) + 5: buffer(a, h, cap * .9, 650)
        thr = get('graph_now.osat.thr_eff', [self.nodes[n]['osat']['thr'] for n in self.oi])
        for n, oi in self.oi.items():
            os = self.nodes[n]['osat']
            for h in range(H):
                terms = {}
                for raw, pk in os['packages'].items():
                    a = self.si[n, self.kid[raw]]; b = self.si[n, self.kid[pk]]
                    credit = self.salvage[a] if remaining == H and h + int(os['tau']) >= H else 0
                    x = var(-credit); terms[x] = 1; term(a, h, x, 1); term(b, h + int(os['tau']), x, -1)
                row(ub, br, terms, max(0, thr[oi]))
        forecast = get('demand_forecast.qty', np.zeros((len(self.l['demands']), 1)))
        backlog = get('backlog.qty', np.zeros(len(self.l['demands'])))
        for di, (n, k) in enumerate(self.l['demands']):
            s = self.si[n, k]; back = bool(self.s['sinks']['backlog'][di]); prev = None; penalty = float(self.s['sinks']['pi'][di])
            for h in range(H):
                demand = max(0, forecast[di, min(h, forecast.shape[1] - 1)])
                served = var(0, None if back else demand); missing = var(penalty)
                term(s, h, served, 1); terms = {served: 1, missing: 1}
                if back and prev is not None: terms[prev] = -1
                row(eq, er, terms, demand + (backlog[di] if back and h == 0 else 0)); prev = missing
                if remaining - h > 2: buffer(s, h, demand * .8, penalty * .1)
            targets[s] = min(self.storage[s] * .8, np.mean(forecast[di]) * 1.5)
        for s in range(S):
            for h in range(H):
                if h > 0: bal[s][h][I[s][h - 1]] = -1
                row(eq, er, bal[s][h], arrivals[s, h] + (stock[s] if h == 0 else 0))
                terms = draws[s][h].copy()
                if h > 0: terms[I[s][h - 1]] = -1
                if terms: row(ub, br, terms, stock[s] if h == 0 else 0)
            if remaining > H and s in targets: buffer(s, H - 1, targets[s], 250000 if s in fuel_rates else 1500)
        def matrix(rows):
            rr, cc, vv = [], [], []
            for i, terms in enumerate(rows):
                for j, v in terms.items():
                    if v: rr.append(i); cc.append(j); vv.append(v)
            return coo_matrix((vv, (rr, cc)), shape=(len(rows), len(obj))).tocsr()
        result = linprog(obj, A_ub=matrix(ub), b_ub=np.asarray(br), A_eq=matrix(eq), b_eq=np.asarray(er), bounds=bounds, method='highs', options={'time_limit': 8})
        flows = np.zeros(self.na)
        if result.x is not None:
            for j in range(self.na): flows[j] = max(0, float(result.x[X[j][0]]))
        return {'flows': flows}
