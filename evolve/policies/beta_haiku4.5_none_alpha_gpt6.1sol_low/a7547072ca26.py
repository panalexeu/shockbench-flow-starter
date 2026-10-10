# 0.552033794649389
import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        s = self.s
        self.edges = s['edges']
        self.routes = []
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        a = s['action_slots']
        for i,(e,k,lane) in enumerate(zip(a['edge'],a['k'],a['lane'])):
            path = list(s['lanes']['edges'][lane]) if lane is not None and lane >= 0 else [e]
            cps = list(s['lanes']['chokepoints'][lane]) if lane is not None and lane >= 0 else []
            self.routes.append((i,e,k,lane,path,cps,self.edges['tail'][path[0]],self.edges['head'][path[-1]]))
        self.dest = {(r[7],r[2]) for r in self.routes}
        self.stockslots = self.l['stock_slots']
        self.demands = {tuple(x):i for i,x in enumerate(self.l['demands'])}
        groups = {}
        incoming = {}
        for r in self.routes:
            i,e,k,lane,path,cps,tail,head = r
            cap = min(float(self.edges['u0'][p] or 0) for p in path)
            groups[(tail,head,k)] = max(groups.get((tail,head,k),0),cap)
            incoming[(head,k)] = max(incoming.get((head,k),0),cap)
        self.rate = {}
        for (tail,head,k),cap in groups.items():
            self.rate[(tail,k)] = self.rate.get((tail,k),0) + cap
        for key in self.dest:
            if self.rate.get(key,0) <= 0:
                self.rate[key] = incoming.get(key,0)
        self.override = np.zeros(config['spaces']['action']['override_qty']['shape'])
        self.release = np.zeros(config['spaces']['action']['release_mode']['shape'],dtype=np.int64)

    def act(self, o):
        week = int(o['week'][0])
        remaining = max(0,self.T-week+1)
        stock = {tuple(key):float(q) for key,q in zip(self.stockslots,o['stock.qty'])}
        available = dict(stock)
        for key,q in zip(self.l['supply_slots'],o['graph_now.supply.avail']):
            key = tuple(key)
            available[key] = available.get(key,0)+float(q)
        position = dict(stock)
        live = o.get('pipeline.qty.observed',np.ones_like(o['pipeline.qty']))
        for j in np.flatnonzero(live):
            q = float(o['pipeline.qty'][j])
            if q <= 0: continue
            e = int(o['pipeline.edge'][j]); k = int(o['pipeline.k'][j])
            head = self.edges['head'][e]
            lane = int(o['pipeline.lane'][j])
            lm = o.get('pipeline.lane.observed')
            if (lm is None or lm[j]) and 0 <= lane < len(self.s['lanes']['edges']):
                head = self.edges['head'][self.s['lanes']['edges'][lane][-1]]
            key = (head,k)
            position[key] = position.get(key,0)+q
        if 'queue_lots.qty' in o:
            for key,row in zip(self.l.get('lot_keys',[]),o['queue_lots.qty']):
                cp,k,lane,e = key
                head = self.edges['head'][e]
                if lane is not None and lane >= 0:
                    head = self.edges['head'][self.s['lanes']['edges'][lane][-1]]
                pair = (head,k)
                position[pair] = position.get(pair,0)+float(np.sum(row))
        if 'wip.qty' in o:
            mask = o.get('wip.qty.observed',np.ones_like(o['wip.qty']))
            for j in np.flatnonzero(mask):
                pair = (int(o['wip.node'][j]),int(o['wip.k'][j]))
                position[pair] = position.get(pair,0)+float(o['wip.qty'][j])
        forecast = o['demand_forecast.qty']
        rates = dict(self.rate)
        for pair,j in self.demands.items():
            rates[pair] = float(np.mean(forecast[j,:min(4,forecast.shape[1])]))
        u = o['graph_now.u']; costs = o['graph_now.c']; tau = o['graph_now.tau']
        tariffs = o['graph_now.tariff']
        values = self.s['commodities']['v']
        options = []
        minlead = {}
        for r in self.routes:
            i,e,k,lane,path,cps,tail,head = r
            if not o['action_mask'][i]: continue
            cap = min(float(u[p]) for p in path)
            lead = sum(max(0,int(tau[p])) for p in path)
            closure = 0.0
            for cp in cps:
                ix = self.cp.get(cp)
                if ix is not None:
                    opened = float(o['graph_now.open'][ix])
                    closure += 8*(1-opened)
                    pool = self.s['commodities']['pool'][k]
                    field = 'graph_now.kappa.' + str(pool)
                    if field in o:
                        cap = min(cap,float(o[field][ix])*opened)
            if cap <= 0 or lead >= remaining: continue
            pair = (head,k)
            minlead[pair] = min(minlead.get(pair,1e9),lead+closure)
            expense = sum(float(costs[p])+float(tariffs[p,k])*float(values[k]) for p in path)
            score = expense/max(float(values[k]),1.0) + 0.035*lead + 0.18*closure
            options.append((score,r,cap,lead))
        flows = np.zeros(len(self.routes),dtype=float)
        edge_used = np.zeros(len(u))
        targets = {}
        for pair in self.dest:
            lead = minlead.get(pair,0)
            horizon = min(remaining,max(1.0,lead+1.6))
            target = rates.get(pair,0)*horizon
            if pair in self.demands:
                j = self.demands[pair]
                n = min(forecast.shape[1],max(1,int(np.ceil(horizon))))
                target = float(np.sum(forecast[j,:n])) + max(0,horizon-n)*rates[pair]
                target += float(o['backlog.qty'][j])
            targets[pair] = target
        for score,r,cap,lead in sorted(options,key=lambda x:x[0]):
            i,e,k,lane,path,cps,tail,head = r
            pair = (head,k); origin = (tail,k)
            deficit = max(0,targets.get(pair,0)-position.get(pair,0))
            room = min(max(0,float(u[p])-edge_used[p]) for p in path)
            qty = min(cap,room,deficit,max(0,available.get(origin,0)))
            if qty <= 0: continue
            flows[i] = qty
            available[origin] = available.get(origin,0)-qty
            position[pair] = position.get(pair,0)+qty
            for p in path: edge_used[p] += qty
        return {'flows':flows,'override_qty':self.override,'release_mode':self.release}
