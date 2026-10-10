# 0.5605474658320913
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
        self.override = np.zeros(config['spaces']['action']['override_qty']['shape'])
        self.release = np.zeros(config['spaces']['action']['release_mode']['shape'],dtype=np.int64)
        self.tanker_commodities = set(i for i,o in enumerate(self.s['commodities']['override']) if o)

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
        warning = o.get('warning.score',np.zeros(1))
        max_warn = float(np.max(warning)) if len(warning) > 0 else 0.0
        disruption_intensity = max_warn
        pending = {}
        if 'pending_prohibitions.edge' in o:
            mask = o.get('pending_prohibitions.edge.observed',np.ones_like(o['pending_prohibitions.edge']))
            for j in np.flatnonzero(mask):
                p = (int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
                w = int(o['pending_prohibitions.effective_week'][j])
                if w >= week:
                    pending[p] = min(pending.get(p,w),w)
        closure_severity = 0.0
        if 'closure_end.chokepoint' in o:
            mask = o.get('closure_end.chokepoint.observed',np.ones_like(o['closure_end.chokepoint']))
            for j in np.flatnonzero(mask):
                end = int(o['closure_end.end_week'][j]) if o.get('closure_end.end_week.observed',[0])[0] else self.T+10
                if end > week:
                    closure_severity += 1.0
        options = []
        minlead = {}
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
                    closure += 8.0*(1-op)
                    field = 'graph_now.kappa.'+str(self.s['commodities']['pool'][k])
                    if field in o:
                        pool_constraint = min(pool_constraint,float(o[field][ix])*op)
                    war += 0.35*int(o['graph_now.war_risk'][ix])
            if pool_constraint < 1e9:
                cap = min(cap,pool_constraint)
            if cap <= 0 or lead >= remaining:
                continue
            pair = (head,k)
            minlead[pair] = min(minlead.get(pair,1e9),lead+closure)
            value = float(self.s['commodities']['v'][k])
            expense = sum(float(c[p])+float(o['graph_now.tariff'][p,k])*value for p in path)
            elapsed,risk = 0,0.0
            for p in path:
                if pending.get((p,k),self.T+100) <= week+elapsed:
                    risk += 2.8
                elapsed += max(0,int(tau[p]))
            base_score = expense/max(value,1)+0.032*lead+0.19*closure+war+risk
            warn_mult = 1.0+0.30*disruption_intensity
            score = base_score*warn_mult
            options.append((score,r,cap,lead,pair))
        live = o.get('pipeline.qty.observed',np.ones_like(o['pipeline.qty']))
        for j in np.flatnonzero(live):
            q = float(o['pipeline.qty'][j])
            if q <= 0:
                continue
            e,k = int(o['pipeline.edge'][j]),int(o['pipeline.k'][j])
            head = self.e['head'][e]
            lane = int(o['pipeline.lane'][j])
            lm = o.get('pipeline.lane.observed')
            arrival = int(o['pipeline.arrival_week'][j])
            delay = max(0,arrival-week)
            if (lm is None or lm[j]) and 0 <= lane < len(self.s['lanes']['edges']):
                path = self.s['lanes']['edges'][lane]
                head = self.e['head'][path[-1]]
                if e in path:
                    delay += sum(max(0,int(tau[p])) for p in path[path.index(e)+1:])
                for cp in self.s['lanes']['chokepoints'][lane]:
                    ix = self.cp.get(cp)
                    if ix is not None:
                        delay += 6.0*(1-float(opened[ix]))
            pair = (head,k)
            if delay >= remaining:
                weight = 0.0
            else:
                pair_options = [sc for sc in options if sc[4]==pair]
                if pair_options:
                    min_lead = min(sc[3] for sc in pair_options)
                    horizon = min(remaining,max(1.5,min_lead+2.5+1.5*disruption_intensity))
                else:
                    horizon = min(remaining,max(1.5,2.5+1.5*disruption_intensity))
                weight = min(1.0,max(0.12,(remaining-delay)/max(horizon,0.5)))
            position[pair] = position.get(pair,0)+q*weight
        for key,row in zip(self.l.get('lot_keys',[]),o.get('queue_lots.qty',[])):
            cp,k,lane,e = key
            head = self.e['head'][e]
            if lane is not None and lane >= 0:
                head = self.e['head'][self.s['lanes']['edges'][lane][-1]]
            ix = self.cp.get(cp)
            op = float(opened[ix]) if ix is not None else 1.0
            weight = 0.20+0.80*op
            pair = (head,k)
            position[pair] = position.get(pair,0)+float(np.sum(row))*weight
        if 'wip.qty' in o:
            mask = o.get('wip.qty.observed',np.ones_like(o['wip.qty']))
            for j in np.flatnonzero(mask):
                out_week = int(o['wip.out_week'][j])
                if out_week <= self.T:
                    pair = (int(o['wip.node'][j]),int(o['wip.k'][j]))
                    position[pair] = position.get(pair,0)+float(o['wip.qty'][j])
        forecast = o['demand_forecast.qty']
        targets = {}
        for pair in self.dest:
            options_for_pair = [sc for sc in options if sc[4]==pair]
            if options_for_pair:
                lead_val = min(sc[3] for sc in options_for_pair)
                horizon = min(remaining,max(1.5,lead_val+2.5+1.5*disruption_intensity))
            else:
                horizon = min(remaining,max(1.5,2.5+1.5*disruption_intensity))
            rate = self.rate.get(pair,0)
            target = rate*horizon
            if pair in self.demands:
                j = self.demands[pair]
                n_forecast = min(len(forecast[j]),max(2,int(np.ceil(horizon))+1))
                target = float(np.sum(forecast[j,:n_forecast]))
                if horizon > n_forecast and n_forecast < len(forecast[j]):
                    tail_avg = float(np.mean(forecast[j,max(0,n_forecast-2):n_forecast]))
                    target += (horizon-n_forecast)*tail_avg
                target += float(o['backlog.qty'][j])
                if self.backlog.get(pair,False):
                    base_mult = 1.15+0.08*disruption_intensity
                    target *= base_mult
                    target += 0.12*self.sinks_pi.get(pair,0)
            targets[pair] = target*(1.0+0.14*disruption_intensity)
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
        return {'flows':flows,'override_qty':self.override,'release_mode':self.release}