# -1.4332731008460247
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
        self.nid = {x: i for i, x in enumerate(self.s['nodes']['id'])}
        self.kid = {x: i for i, x in enumerate(self.s['commodities']['id'])}
        self.slots = [tuple(x) for x in self.l['stock_slots']]
        self.si = {x: i for i, x in enumerate(self.slots)}
        self.cp = {n: i for i, n in enumerate(self.l['chokepoints'])}
        self.ford = {n: i for i, n in enumerate(self.l['fabs'])}
        self.gord = {n: i for i, n in enumerate(self.l['grids'])}
        self.oord = {n: i for i, n in enumerate(self.l['osats'])}
        self.ns = len(self.slots)
        self.na = len(self.s['action_slots']['edge'])
        self.storage = []
        self.hold = []
        for n, k in self.slots:
            a = self.nodes[n].get('stock', {}).get(self.s['commodities']['id'][k], {})
            self.storage.append(float(a.get('storage', 1e12) or 1e12))
            self.hold.append(float(a.get('holding_cost', 0)))
        self.routes = []
        for j in range(self.na):
            e = self.s['action_slots']['edge'][j]
            ln = self.s['action_slots']['lane'][j]
            es = list(self.s['lanes']['edges'][ln]) if ln is not None and ln >= 0 else [e]
            self.routes.append((e, self.s['action_slots']['k'][j], ln, es, self.s['edges']['tail'][e], self.s['edges']['head'][es[-1]]))
        self.last = {}

    def act(self, observation):
        o = observation
        def get(key, default):
            a = np.asarray(o.get(key, default)).copy()
            mask = o.get(key + '.observed')
            if mask is not None and np.shape(mask) == a.shape:
                d = self.last.get(key, np.asarray(default))
                a = np.where(mask, a, d)
            self.last[key] = a.copy()
            return a
        week = int(o['week'][0])
        H = min(16, int(self.c['T']) - week + 1)
        S = self.ns
        stock = get('stock.qty', np.zeros(S))
        tau = get('graph_now.tau', self.s['edges']['tau0']).astype(int)
        cost = get('graph_now.c', self.s['edges']['c0'])
        cap = get('graph_now.u', [x if x is not None else 0 for x in self.s['edges']['u0']])
        prohib = get('graph_now.prohibited', np.zeros((len(cap), len(self.kid))))
        tariff = get('graph_now.tariff', np.zeros_like(prohib, dtype=float))
        opens = get('graph_now.open', np.ones(len(self.cp)))
        arrivals = np.zeros((S, H))
        def route_remaining(e, ln):
            if ln is not None and ln >= 0:
                es = self.s['lanes']['edges'][int(ln)]
                if e in es:
                    return es[es.index(e) + 1:]
            return []
        def add_arr(n, k, q, t):
            if (n, k) in self.si and 0 <= t < H:
                arrivals[self.si[n, k], int(t)] += q
        pq = np.asarray(o.get('pipeline.qty', []))
        pm = np.asarray(o.get('pipeline.qty.observed', pq > 0))
        for z in np.flatnonzero(pm):
            e = int(o['pipeline.edge'][z]); k = int(o['pipeline.k'][z])
            ln = int(o['pipeline.lane'][z]) if o.get('pipeline.lane.observed', np.ones(len(pq)))[z] else None
            rest = route_remaining(e, ln)
            delay = sum(max(1, int(tau[x])) for x in rest)
            for x in rest:
                n = self.s['edges']['tail'][x]
                if n in self.cp and opens[self.cp[n]] < .1:
                    delay += 5
            dst = self.s['edges']['head'][rest[-1] if rest else e]
            add_arr(dst, k, float(pq[z]), int(o['pipeline.arrival_week'][z]) - week + delay)
        if 'queue_lots.qty' in o:
            for z, key in enumerate(self.l.get('lot_keys', [])):
                n, k, ln, e = key
                q = float(np.sum(o['queue_lots.qty'][z]))
                es = [e] + route_remaining(e, ln)
                delay = sum(max(1, int(tau[x])) for x in es)
                if opens[self.cp[n]] < .1:
                    delay += 5
                add_arr(self.s['edges']['head'][es[-1]], k, q, delay)
        for z, q in enumerate(o.get('wip.qty', [])):
            if q > 0:
                add_arr(int(o['wip.node'][z]), int(o['wip.k'][z]), q, int(o['wip.out_week'][z]) - week)
        obj = []; bounds = []; eq = []; rhs = []; ub = []; urhs = []
        def var(c=0, hi=None):
            j = len(obj); obj.append(c / 10000.0); bounds.append((0, hi)); return j
        def row(table, rr, terms, b):
            table.append(terms); rr.append(float(b))
        I = [[var(self.hold[s], self.storage[s]) for h in range(H)] for s in range(S)]
        bal = [[{I[s][h]: 1.0} for h in range(H)] for s in range(S)]
        dispatch = [[{} for h in range(H)] for s in range(S)]
        targets = np.zeros(S)
        def term(s, h, j, v):
            if s is not None and 0 <= h < H:
                bal[s][h][j] = bal[s][h].get(j, 0) + v
        X = []
        edgegroups = {}
        for j, (e, k, ln, es, src, dst) in enumerate(self.routes):
            a = self.si.get((src, k)); b = self.si.get((dst, k))
            delay = sum(max(1, int(tau[x])) for x in es)
            blocked = any(prohib[x, k] for x in es)
            congest = False
            for x in es:
                n = self.s['edges']['tail'][x]
                if n in self.cp and opens[self.cp[n]] < .15:
                    congest = True
            price = sum(float(cost[x]) + float(tariff[x, k]) * self.s['commodities']['v'][k] for x in es)
            xs = []
            for h in range(H):
                hi = max(0, float(cap[e]))
                if blocked or congest or a is None or b is None:
                    hi = 0
                x = var(price, hi); xs.append(x)
                term(a, h, x, 1); term(b, h + delay, x, -1)
                if a is not None:
                    dispatch[a][h][x] = 1
                edgegroups.setdefault((e, h), {})[x] = 1
            X.append(xs)
        for (e, h), terms in edgegroups.items():
            row(ub, urhs, terms, max(0, cap[e]))
        avail = get('graph_now.supply.avail', np.zeros(len(self.l['supply_slots'])))
        for z, key in enumerate(self.l['supply_slots']):
            s = self.si.get(tuple(key))
            if s is not None:
                for h in range(H):
                    x = var(0, max(0, float(avail[z]))); term(s, h, x, -1)
        fabvars = {}
        R = get('graph_now.fab.R', np.ones(len(self.ford)))
        alpha = get('graph_now.fab.alpha_bar', np.ones(len(self.ford)))
        for n, fi in self.ford.items():
            f = self.nodes[n]['fab']; a = self.si[n, self.kid[f['input']]]; b = self.si[n, self.kid[f['product']]]
            xs = []
            for h in range(H):
                x = var(0, max(0, float(f['cap0']) * float(R[fi]) * float(alpha[fi])))
                term(a, h, x, 1); term(b, h + int(f['tau']), x, -1); xs.append(x)
            fabvars[n] = xs
            targets[a] = float(f['cap0']) * 2
        thr = get('graph_now.osat.thr_eff', [self.nodes[n]['osat']['thr'] for n in self.oord])
        for n, oi in self.oord.items():
            os = self.nodes[n]['osat']
            for h in range(H):
                terms = {}
                for raw, pk in os['packages'].items():
                    a = self.si[n, self.kid[raw]]; b = self.si[n, self.kid[pk]]
                    x = var(); term(a, h, x, 1); term(b, h + int(os['tau']), x, -1); terms[x] = 1
                row(ub, urhs, terms, max(0, thr[oi]))
        G = get('graph_now.grid.G_bar', [self.nodes[n]['grid']['deliverable'] for n in self.gord])
        Y = get('graph_now.grid.y_bar', [self.nodes[n]['grid']['base_load'] for n in self.gord])
        psi = float(self.raw['params'].get('psi', 0))
        for n, gi in self.gord.items():
            g = self.nodes[n]['grid']
            for h in range(H):
                terms = {}
                for fuel, share in g['shares'].items():
                    if fuel not in self.kid:
                        continue
                    k = self.kid[fuel]; s = self.si.get((n, k))
                    if s is None: continue
                    mx = max(0, float(share) * G[gi])
                    if h == 0 and fuel == g.get('rationed'):
                        threshold = psi * float(g.get('ibar', {}).get(fuel, 0))
                        if threshold > 0: mx *= min(1, stock[s] / threshold)
                    burn = var(0, mx); term(s, h, burn, 1); terms[burn] = -1
                    targets[s] = max(float(g.get('ibar', {}).get(fuel, 0)), float(share) * G[gi] * 3)
                shed = var(float(g.get('voll', 4100000)), max(0, Y[gi])); terms[shed] = -1
                for fn, xs in fabvars.items():
                    f = self.nodes[fn]['fab']
                    if f.get('grid') == self.s['nodes']['id'][n]:
                        terms[xs[h]] = float(f['e']) / max(.001, R[self.ford[fn]])
                unmodel = float(g['shares'].get('unmodelled', 0)) * G[gi]
                row(ub, urhs, terms, unmodel - Y[gi])
        forecast = get('demand_forecast.qty', np.zeros((len(self.l['demands']), 1)))
        backlog = get('backlog.qty', np.zeros(len(self.l['demands'])))
        for di, (n, k) in enumerate(self.l['demands']):
            s = self.si[n, k]
            penalty = float(self.s['sinks']['pi'][di])
            back = bool(self.s['sinks']['backlog'][di])
            previous = None
            for h in range(H):
                d = max(0, float(forecast[di, min(h, forecast.shape[1] - 1)]))
                served = var(0, None if back else d); term(s, h, served, 1)
                missing = var(penalty)
                terms = {served: 1, missing: 1}
                if back and previous is not None: terms[previous] = -1
                row(eq, rhs, terms, d + (float(backlog[di]) if back and h == 0 else 0))
                previous = missing
            targets[s] = float(np.mean(forecast[di])) * 1.5
        for s in range(S):
            for h in range(H):
                if h > 0: bal[s][h][I[s][h - 1]] = -1
                row(eq, rhs, bal[s][h], arrivals[s, h] + (stock[s] if h == 0 else 0))
                terms = dispatch[s][h].copy()
                if h > 0: terms[I[s][h - 1]] = -1
                if terms: row(ub, urhs, terms, stock[s] if h == 0 else 0)
            if targets[s] > 0 and week + H - 1 < int(self.c['T']):
                deficit = var(1000 if self.s['nodes']['type'][self.slots[s][0]] != 'grid' else 100000)
                row(ub, urhs, {I[s][-1]: -1, deficit: -1}, -min(targets[s], self.storage[s] * .8))
        def matrix(rows):
            rr = []; cc = []; vv = []
            for i, r in enumerate(rows):
                for j, v in r.items(): rr.append(i); cc.append(j); vv.append(v)
            return coo_matrix((vv, (rr, cc)), shape=(len(rows), len(obj))).tocsr()
        result = linprog(obj, A_ub=matrix(ub), b_ub=urhs, A_eq=matrix(eq), b_eq=rhs, bounds=bounds, method='highs', options={'time_limit': 8})
        flows = np.zeros(self.na)
        if result.x is not None:
            for j in range(self.na): flows[j] = max(0, result.x[X[j][0]])
        return {'flows': flows}
