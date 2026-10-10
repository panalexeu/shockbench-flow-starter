# 0.47785280823071163
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        s = self.s
        self.tail = np.asarray(s['edges']['tail'], dtype=int)
        self.head = np.asarray(s['edges']['head'], dtype=int)
        self.u0 = np.asarray([0 if x is None else x for x in s['edges']['u0']], dtype=float)
        self.c0 = np.asarray(s['edges']['c0'], dtype=float)
        self.tau0 = np.asarray(s['edges']['tau0'], dtype=float)
        self.values = np.asarray(s['commodities']['v'], dtype=float)
        self.types = s['nodes']['type']
        self.routes = []
        self.groups = {}
        self.stock_index = {tuple(p): i for i, p in enumerate(self.l['stock_slots'])}
        self.demand_index = {tuple(p): i for i, p in enumerate(self.l['demands'])}
        self.cp_index = {n:i for i,n in enumerate(self.l['chokepoints'])}
        a = s['action_slots']
        for i, (e,k,lane) in enumerate(zip(a['edge'], a['k'], a['lane'])):
            es = [e] if lane is None or lane < 0 else list(s['lanes']['edges'][lane])
            cps = [] if lane is None or lane < 0 else list(s['lanes']['chokepoints'][lane])
            r = (int(e), int(k), lane, es, cps, int(self.tail[e]), int(self.head[es[-1]]))
            self.routes.append(r)
            self.groups.setdefault((r[5],k,r[6]), []).append(i)
        self.rate = {}
        for (src,k,dst), ids in self.groups.items():
            canonical = []
            for i in ids:
                e,_,lane,es,_,_,_ = self.routes[i]
                alt = s['edges']['alt_of'][e] if lane is None or lane < 0 else s['lanes']['alt_of'][lane]
                if alt is None:
                    canonical.append(i)
            ids0 = canonical or [min(ids, key=lambda i: sum(self.c0[e] for e in self.routes[i][3]))]
            cap = sum(min(self.u0[e] for e in self.routes[i][3]) for i in ids0)
            self.rate[(dst,k)] = self.rate.get((dst,k),0.) + cap
        self.last = {}

    def field(self, o, name, fallback):
        if name not in o:
            return np.asarray(fallback).copy()
        x = np.asarray(o[name])
        mask = o.get(name+'.observed')
        if mask is None or np.shape(mask) != x.shape:
            return x.copy()
        return np.where(mask, x, fallback)

    def act(self, o):
        week = int(o['week'][0])
        u = self.field(o, 'graph_now.u', self.u0).astype(float)
        c = self.field(o, 'graph_now.c', self.c0)
        tau = self.field(o, 'graph_now.tau', self.tau0)
        tariff = np.asarray(o.get('graph_now.tariff', np.zeros((len(u),len(self.values)))))
        opened = self.field(o, 'graph_now.open', np.ones(len(self.cp_index)))
        mask = np.asarray(o.get('action_mask', np.ones(len(self.routes))))
        stock = self.field(o, 'stock.qty', np.zeros(len(self.stock_index)))
        inventory = {p: max(0.,float(stock[i])) for p,i in self.stock_index.items()}
        available = dict(inventory)
        supply = self.field(o, 'graph_now.supply.avail', np.zeros(len(self.l['supply_slots'])))
        for i,p in enumerate(self.l['supply_slots']):
            p = tuple(p)
            available[p] = available.get(p,0.) + max(0.,float(supply[i]))
        forecast = np.asarray(o['demand_forecast.qty'])
        backlog = np.asarray(o.get('backlog.qty', np.zeros(len(self.l['demands']))))
        pipeqty = np.asarray(o.get('pipeline.qty', []))
        live = np.asarray(o.get('pipeline.qty.observed', np.ones(len(pipeqty))))
        for j in np.flatnonzero(live):
            q = float(pipeqty[j])
            if q <= 0:
                continue
            e = int(o['pipeline.edge'][j])
            k = int(o['pipeline.k'][j])
            lane = int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed', np.ones(len(pipeqty)))[j] else -1
            dst = int(self.head[e])
            if lane >= 0 and lane < len(self.s['lanes']['edges']):
                dst = int(self.head[self.s['lanes']['edges'][lane][-1]])
            inventory[(dst,k)] = inventory.get((dst,k),0.) + q
        if 'queue_lots.qty' in o:
            for j,key in enumerate(self.l.get('lot_keys', [])):
                cp,k,lane,e = key
                dst = int(self.head[e])
                if lane is not None and lane >= 0:
                    dst = int(self.head[self.s['lanes']['edges'][lane][-1]])
                inventory[(dst,k)] = inventory.get((dst,k),0.) + float(np.sum(o['queue_lots.qty'][j]))
        for j in np.flatnonzero(o.get('wip.qty.observed', [])):
            p = (int(o['wip.node'][j]), int(o['wip.k'][j]))
            inventory[p] = inventory.get(p,0.) + float(o['wip.qty'][j])
        pending = {}
        if 'pending_prohibitions.edge' in o:
            for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed', [])):
                p = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
                pending[p] = min(pending.get(p,self.T+100),int(o['pending_prohibitions.effective_week'][j]))
        candidates = []
        for i,r in enumerate(self.routes):
            e,k,lane,es,cps,src,dst = r
            if not mask[i]:
                continue
            if any(opened[self.cp_index[cp]] <= 0.01 for cp in cps if cp in self.cp_index):
                continue
            cap = min(max(0.,u[x]) for x in es)
            if cap <= 0:
                continue
            lead = sum(max(1.,tau[x]) for x in es)
            if week + lead > self.T + 1:
                continue
            cost = sum(c[x] + tariff[x,k]*self.values[k] for x in es)
            risk = 0.
            elapsed = 0.
            for x in es:
                if pending.get((x,k),self.T+100) <= week + elapsed:
                    risk += 1e9
                elapsed += max(1.,tau[x])
            score = cost + .002*self.values[k]*lead + risk
            candidates.append((score,i,cap,lead))
        flows = np.zeros(len(self.routes), dtype=float)
        remaining = np.maximum(0.,u.copy())
        for score,i,cap,lead in sorted(candidates):
            e,k,lane,es,cps,src,dst = self.routes[i]
            p = (dst,k)
            if p in self.demand_index:
                d = self.demand_index[p]
                row = forecast[d]
                h = min(len(row), max(1,int(np.ceil(lead))+2),max(0,self.T-week+1))
                target = float(np.sum(row[:h])) + float(backlog[d])
            else:
                rate = self.rate.get(p,cap)
                kind = self.types[dst]
                buffer = 3. if kind in ('fab','osat','material') else 2.
                target = rate * min(lead + buffer,max(0,self.T-week+1))
            need = max(0.,target-inventory.get(p,0.))
            have = available.get((src,k),0.)
            q = min(need,have,cap,min(remaining[x] for x in es))
            if q <= 0 or score >= 1e9:
                continue
            flows[i] = q
            available[(src,k)] = have-q
            inventory[p] = inventory.get(p,0.)+q
            for x in es:
                remaining[x] = max(0.,remaining[x]-q)
        return {'flows': flows}
