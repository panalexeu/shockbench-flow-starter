# 0.566565473043425

import numpy as np

class Agent:
    def __init__(self, config):
        self.s = config['static']
        self.l = config['layout']
        self.T = config['T']
        self.e = self.s['edges']
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        self.routes = []
        a = self.s['action_slots']
        groups, incoming = {}, {}
        for i,(e,k,lane) in enumerate(zip(a['edge'],a['k'],a['lane'])):
            path = list(self.s['lanes']['edges'][lane]) if lane is not None and lane >= 0 else [e]
            cps = list(self.s['lanes']['chokepoints'][lane]) if lane is not None and lane >= 0 else []
            tail,head = self.e['tail'][path[0]],self.e['head'][path[-1]]
            self.routes.append((i,k,lane,path,cps,tail,head))
            cap = min(float(self.e['u0'][p] or 0) for p in path)
            groups[(tail,head,k)] = max(groups.get((tail,head,k),0),cap)
            incoming[(head,k)] = max(incoming.get((head,k),0),cap)
        self.dest = {(r[6],r[1]) for r in self.routes}
        self.rate = {}
        for (tail,head,k),cap in groups.items():
            self.rate[(tail,k)] = self.rate.get((tail,k),0)+cap
        for pair in self.dest:
            if self.rate.get(pair,0) <= 0:
                self.rate[pair] = incoming.get(pair,0)
        self.demands = {tuple(p):j for j,p in enumerate(self.l['demands'])}
        self.backlog = {(n,k):bool(b) for n,k,b in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['backlog'])}
        self.sinks_pi = {(n,k):float(pi) for n,k,pi in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['pi'])}
        self.sinks_node_k = {(n,k):i for i,(n,k) in enumerate(zip(self.s['sinks']['node'],self.s['sinks']['k']))}

    def act(self,o):
        week = int(o['week'][0])
        remaining = max(0,self.T-week+1)
        stock = {tuple(p):float(q) for p,q in zip(self.l['stock_slots'],o['stock.qty'])}
        available = dict(stock)
        for p,q in zip(self.l['supply_slots'],o['graph_now.supply.avail']):
            p = tuple(p)
            available[p] = available.get(p,0)+float(q)
        position = dict(stock)
        u,c,tau = o['graph_now.u'],o['graph_now.c'],o['graph_now.tau']
        opened = o['graph_now.open']
        pending = {}
        if 'pending_prohibitions.edge' in o:
            mask = o.get('pending_prohibitions.edge.observed',np.ones_like(o['pending_prohibitions.edge']))
            for j in np.flatnonzero(mask):
                p = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
                w = int(o['pending_prohibitions.effective_week'][j])
                if w >= week:
                    pending[p] = min(pending.get(p,w),w)
        warning_scores = o.get('warning.score', np.zeros(1))
        max_warning = float(np.max(warning_scores)) if len(warning_scores) > 0 else 0.0
        high_alert = max_warning > 0.65
        options, minlead, preferred = [], {}, {}
        for r in self.routes:
            i,k,lane,path,cps,tail,head = r
            if not o['action_mask'][i]:
                continue
            cap = min(float(u[p]) for p in path)
            lead = sum(max(0,int(tau[p])) for p in path)
            closure,war = 0.0,0.0
            pool_constraint = 1e9
            for cp in cps:
                ix = self.cp.get(cp)
                if ix is not None:
                    op = float(opened[ix])
                    closure += 8.6*(1-op)
                    field = 'graph_now.kappa.'+str(self.s['commodities']['pool'][k])
                    if field in o:
                        pool_constraint = min(pool_constraint,float(o[field][ix])*op)
                    war += 0.38*int(o['graph_now.war_risk'][ix])
            if pool_constraint < 1e9:
                cap = min(cap,pool_constraint)
            if cap <= 0 or lead >= remaining:
                continue
            elapsed, blocked = 0, False
            for p in path:
                if pending.get((p,k),self.T+100) <= week+elapsed:
                    blocked = True
                    break
                elapsed += max(0,int(tau[p]))
            if blocked:
                continue
            pair = (head,k)
            minlead[pair] = min(minlead.get(pair,1e9),lead+closure)
            value = float(self.s['commodities']['v'][k])
            expense = sum(float(c[p])+float(o['graph_now.tariff'][p,k])*value for p in path)
            score = expense/max(value,1)+0.035*lead+0.20*closure+war
            options.append((score,r,cap,lead,pair))
            if pair not in preferred or score < preferred[pair][0]:
                preferred[pair] = (score, lead+closure)
        coverage = {p:v[1]+2.5 for p,v in preferred.items()}
        live = o.get('pipeline.qty.observed',np.ones_like(o['pipeline.qty']))
        for j in np.flatnonzero(live):
            q = float(o['pipeline.qty'][j])
            if q <= 0:
                continue
            e,k = int(o['pipeline.edge'][j]),int(o['pipeline.k'][j])
            head = self.e['head'][e]
            lane = int(o['pipeline.lane'][j])
            lm = o.get('pipeline.lane.observed')
            delay = max(0,int(o['pipeline.arrival_week'][j])-week)
            if (lm is None or lm[j]) and 0 <= lane < len(self.s['lanes']['edges']):
                path = self.s['lanes']['edges'][lane]
                head = self.e['head'][path[-1]]
                if e in path:
                    delay += sum(max(0,int(tau[p])) for p in path[path.index(e)+1:])
                for cp in self.s['lanes']['chokepoints'][lane]:
                    ix = self.cp.get(cp)
                    if ix is not None:
                        delay += 6.8*(1-float(opened[ix]))
            pair = (head,k)
            horizon = coverage.get(pair,delay+2.5)
            weight = min(1.0,max(0.12,horizon/max(delay,1)))
            if delay >= remaining:
                weight = 0.02
            position[pair] = position.get(pair,0)+q*weight
        for key,row in zip(self.l.get('lot_keys',[]),o.get('queue_lots.qty',[])):
            cp,k,lane,e = key
            head = self.e['head'][e]
            if lane is not None and lane >= 0:
                head = self.e['head'][self.s['lanes']['edges'][lane][-1]]
            ix = self.cp.get(cp)
            op = float(opened[ix]) if ix is not None else 1.0
            weight = 0.18+0.82*op
            pair = (head,k)
            position[pair] = position.get(pair,0)+float(np.sum(row))*weight
        if 'wip.qty' in o:
            mask = o.get('wip.qty.observed',np.ones_like(o['wip.qty']))
            for j in np.flatnonzero(mask):
                pair = (int(o['wip.node'][j]),int(o['wip.k'][j]))
                if int(o['wip.out_week'][j]) <= self.T:
                    position[pair] = position.get(pair,0)+float(o['wip.qty'][j])
        forecast = o['demand_forecast.qty']
        targets = {}
        alert_mult = 1.28 if high_alert else 1.12
        for pair in self.dest:
            horizon = min(remaining,max(1.0,coverage.get(pair,2.5)))
            rate = self.rate.get(pair,0)
            target = rate*horizon
            if pair in self.demands:
                j = self.demands[pair]
                rate = float(np.mean(forecast[j,:min(4,forecast.shape[1])]))
                n = min(forecast.shape[1],int(np.floor(horizon)))
                target = float(np.sum(forecast[j,:n]))
                if horizon > n and n < forecast.shape[1]:
                    target += (horizon-n)*float(forecast[j,n])
                elif horizon > n:
                    target += (horizon-n)*rate
                target += float(o['backlog.qty'][j])
                if self.backlog.get(pair,False):
                    target *= 1.22
                    if high_alert:
                        target *= alert_mult
            targets[pair] = target
        flows = np.zeros(len(self.routes))
        used = np.zeros(len(u))
        for score,r,cap,lead,pair in sorted(options,key=lambda x:x[0]):
            i,k,lane,path,cps,tail,head = r
            origin = (tail,k)
            deficit = max(0,targets.get(pair,0)-position.get(pair,0))
            room = min(max(0,float(u[p])-used[p]) for p in path)
            qty = min(cap,room,deficit,max(0,available.get(origin,0)))
            if qty > 1e-6:
                flows[i] = qty
                available[origin] = available.get(origin,0)-qty
                position[pair] = position.get(pair,0)+qty
                for p in path:
                    used[p] += qty
        return {'flows':flows}