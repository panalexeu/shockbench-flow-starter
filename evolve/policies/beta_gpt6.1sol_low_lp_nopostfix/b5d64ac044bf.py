# -1.590894016732159
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Agent:
    def __init__(self, config):
        self.cfg = config
        self.s = config['static']
        self.l = config['layout']
        self.T = int(config['T'])
        self.slots = list(zip(self.s['action_slots']['edge'], self.s['action_slots']['k'], self.s['action_slots']['lane']))
        self.stock = [tuple(x) for x in self.l['stock_slots']]
        self.dem = [tuple(x) for x in self.l['demands']]
        self.types = self.s['nodes']['type']
        self.names = self.s['commodities']['id']
        self.tail = self.s['edges']['tail']
        self.head = self.s['edges']['head']
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        self.routes = []
        for e,k,lane in self.slots:
            es = self.s['lanes']['edges'][int(lane)] if lane is not None and int(lane) >= 0 else [e]
            self.routes.append((self.tail[es[0]], self.head[es[-1]], list(es), k))
        pairs = set(self.stock) | set(self.dem) | set(tuple(x) for x in self.l['supply_slots'])
        for a,b,es,k in self.routes:
            pairs.add((a,k)); pairs.add((b,k))
        self.pairs = sorted(pairs)
        self.pidx = {p:i for i,p in enumerate(self.pairs)}
        self.previous = None

    def act(self, o):
        H = min(10, self.T-int(o['week'][0])+1)
        P = len(self.pairs)
        A = len(self.slots)
        week = int(o['week'][0])
        def observed(key, fallback):
            x = np.asarray(o.get(key, fallback))
            m = np.asarray(o.get(key+'.observed', np.ones_like(x)))
            return np.where(m, x, fallback)
        cap = observed('graph_now.u', np.array([v if v is not None else 0 for v in self.s['edges']['u0']],float))
        tau = observed('graph_now.tau', np.array(self.s['edges']['tau0']))
        cost = observed('graph_now.c', np.array(self.s['edges']['c0']))
        tariff = observed('graph_now.tariff', np.zeros((len(cap),len(self.names))))
        opened = observed('graph_now.open', np.ones(len(self.cp)))
        initial = np.zeros(P)
        for p,q in zip(self.stock, observed('stock.qty', np.zeros(len(self.stock)))):
            initial[self.pidx[p]] += q
        incoming = np.zeros((H,P))
        live = o.get('pipeline.qty.observed', np.ones_like(o['pipeline.qty']))
        for j in np.flatnonzero(live):
            e = int(o['pipeline.edge'][j]); k = int(o['pipeline.k'][j]); q = float(o['pipeline.qty'][j])
            aw = int(o['pipeline.arrival_week'][j]); node = self.head[e]
            lane = int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed',np.ones_like(live))[j] else -1
            if lane >= 0 and lane < len(self.s['lanes']['edges']):
                es = self.s['lanes']['edges'][lane]
                if e in es:
                    pos = es.index(e)
                    aw += sum(max(1,int(tau[x])) for x in es[pos+1:])
                    node = self.head[es[-1]]
            h = max(0,aw-week)
            if h < H and (node,k) in self.pidx:
                incoming[h,self.pidx[node,k]] += q
        for j,q in enumerate(o.get('wip.qty',[])):
            if q <= 0: continue
            p = (int(o['wip.node'][j]),int(o['wip.k'][j]))
            h = max(0,int(o['wip.out_week'][j])-week)
            if p in self.pidx and h < H: incoming[h,self.pidx[p]] += q
        lots = o.get('queue_lots.qty')
        if lots is not None:
            for key,row in zip(self.l.get('lot_keys',[]),lots):
                n,k,lane,e = key
                if lane is not None and lane >= 0:
                    es = self.s['lanes']['edges'][lane]
                    if e not in es: continue
                    rem = es[es.index(e):]
                else: rem = [e]
                h = sum(max(1,int(tau[x])) for x in rem)
                if opened[self.cp[n]] <= 0: h += 3
                p = (self.head[rem[-1]],k)
                if h < H and p in self.pidx: incoming[h,self.pidx[p]] += float(np.sum(row))
        supply = observed('graph_now.supply.avail', np.zeros(len(self.l['supply_slots'])))
        for p,q in zip(self.l['supply_slots'],supply):
            incoming[:,self.pidx[tuple(p)]] += q
        # Automatic production is forecast conservatively; input reserves keep it supplied.
        targets = {}
        def cname(k): return str(self.names[k]).lower()
        fabcap = observed('graph_now.fab.cap_eff',np.zeros(len(self.l['fabs'])))
        oscap = observed('graph_now.osat.thr_eff',np.zeros(len(self.l['osats'])))
        for nodes,caps,kind in [(self.l['fabs'],fabcap,'fab'),(self.l['osats'],oscap,'osat')]:
            for n,c in zip(nodes,caps):
                ins = [k for nn,k in self.pairs if nn==n and ((kind=='fab' and cname(k)=='wafer') or (kind=='osat' and 'raw' in cname(k))) ]
                for k in ins: targets[n,k] = (1.4*c/max(1,len(ins)), 1500.0)
                outs = sorted(set(k for a,b,es,k in self.routes if a==n and ((kind=='fab' and 'raw' in cname(k)) or (kind=='osat' and 'chip' in cname(k) and 'raw' not in cname(k)))))
                for k in outs:
                    for h in range(2,H): incoming[h,self.pidx[n,k]] += .70*c/max(1,len(outs))
        for n in self.l['grids']:
            fuels = [k for nn,k in self.pairs if nn==n and cname(k) in ('lng','crude','nucfuel')]
            for k in fuels:
                p = self.pidx[n,k]
                base = max(initial[p],1.)
                targets[n,k] = (base, 100.0)
                incoming[:,p] -= base/4.0
        forecast = observed('demand_forecast.qty', np.zeros_like(o['demand_forecast.qty']))
        penalties = {(n,k):float(pi) for n,k,pi in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['pi'])}
        backlog = observed('backlog.qty',np.zeros(len(self.dem)))
        c=[]; bounds=[]; eqr=[]; eqc=[]; eqv=[]; ubr=[]; ubc=[]; ubv=[]; rhs=[]
        def var(price,upper=None):
            j=len(c); c.append(float(price)); bounds.append((0,upper)); return j
        inv = np.array([[var(.005) for p in range(P)] for h in range(H)])
        flow = np.full((H,A),-1,int)
        balance = [[{} for p in range(P)] for h in range(H)]
        def add(h,p,j,v): balance[h][p][j]=balance[h][p].get(j,0)+v
        for h in range(H):
            for p in range(P):
                add(h,p,inv[h,p],1)
                if h: add(h,p,inv[h-1,p],-1)
        mask = o.get('action_mask',np.ones(A))
        pending = {}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',np.zeros(0))):
            pending[int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j])] = int(o['pending_prohibitions.effective_week'][j])
        resources = {}
        for h in range(H):
            for a,(src,dst,es,k) in enumerate(self.routes):
                delay = sum(max(1,int(tau[e])) for e in es)
                if h+delay >= H or not mask[a]: continue
                if any(pending.get((e,k),self.T+100)<=week+h for e in es): continue
                routecp = [self.tail[e] for e in es if self.tail[e] in self.cp]
                if any(opened[self.cp[n]] <= 0 for n in routecp): continue
                upper = min(max(0,cap[e]) for e in es)
                price = sum(cost[e]+tariff[e,k]*self.s['commodities']['v'][k] for e in es)
                j=var(price,upper); flow[h,a]=j
                add(h,self.pidx[src,k],j,1); add(h+delay,self.pidx[dst,k],j,-1)
                for e in es: resources.setdefault((h,'edge',e),[]).append(j)
                for n in routecp:
                    pool=self.s['commodities']['pool'][k]
                    resources.setdefault((h,pool,n),[]).append(j)
        for h in range(H):
            for d,p in enumerate(self.dem):
                qty=float(forecast[d,min(h,forecast.shape[1]-1)])
                if h==0: qty+=float(backlog[d])
                j=var(-penalties.get(p,10000),max(0,qty)); add(h,self.pidx[p],j,1)
            for p,(target,penalty) in targets.items():
                j=var(penalty)
                r=len(rhs); rhs.append(-target)
                ubr.extend([r,r]); ubc.extend([inv[h,self.pidx[p]],j]); ubv.extend([-1,-1])
        for (h,typ,n),js in resources.items():
            if typ=='edge': limit=max(0,cap[n])
            else:
                arr=observed('graph_now.kappa.'+typ,np.full(len(self.cp),1e12))
                limit=max(0,arr[self.cp[n]]*opened[self.cp[n]])
            r=len(rhs); rhs.append(limit)
            for j in js: ubr.append(r); ubc.append(j); ubv.append(1)
        beq=[]
        for h in range(H):
            for p in range(P):
                r=len(beq); beq.append(float(incoming[h,p]+(initial[p] if h==0 else 0)))
                # Grid consumption is soft, so low stocks cannot make the LP infeasible.
                if beq[-1]<0:
                    shortage=var(1000); balance[h][p][shortage]=-1
                for j,v in balance[h][p].items(): eqr.append(r); eqc.append(j); eqv.append(v)
        N=len(c)
        ae=coo_matrix((eqv,(eqr,eqc)),shape=(H*P,N)).tocsr()
        au=coo_matrix((ubv,(ubr,ubc)),shape=(len(rhs),N)).tocsr()
        result=linprog(c,A_ub=au,b_ub=rhs,A_eq=ae,b_eq=beq,bounds=bounds,method='highs',options={'time_limit':12.0})
        out=np.zeros(A)
        if result.x is not None:
            for a,j in enumerate(flow[0]):
                if j>=0: out[a]=max(0,float(result.x[j]))
        return {'flows':out}
