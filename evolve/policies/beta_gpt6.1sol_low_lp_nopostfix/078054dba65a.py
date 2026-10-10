# -0.8080400360392034
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

class Agent:
    def __init__(self, config):
        self.s=config['static']; self.l=config['layout']; self.T=int(config['T'])
        self.names=self.s['commodities']['id']; self.tail=self.s['edges']['tail']; self.head=self.s['edges']['head']
        self.cp={n:i for i,n in enumerate(self.l['chokepoints'])}
        self.stock=list(map(tuple,self.l['stock_slots'])); self.dem=list(map(tuple,self.l['demands']))
        self.routes=[]
        for e,k,l in zip(self.s['action_slots']['edge'],self.s['action_slots']['k'],self.s['action_slots']['lane']):
            es=list(self.s['lanes']['edges'][int(l)]) if l is not None and int(l)>=0 else [e]
            self.routes.append((self.tail[es[0]],self.head[es[-1]],es,k))
        pairs=set(self.stock)|set(self.dem)|set(map(tuple,self.l['supply_slots']))
        for a,b,es,k in self.routes: pairs.update([(a,k),(b,k)])
        self.pairs=sorted(pairs); self.idx={p:i for i,p in enumerate(self.pairs)}
        self.production=[]
        for n in self.l['fabs']:
            ins=[k for nn,k in self.pairs if nn==n and self.names[k]=='wafer']
            outs=[k for nn,k in self.pairs if nn==n and 'raw' in self.names[k] and any(a==n and kk==k for a,b,es,kk in self.routes)]
            for k in outs:
                if ins: self.production.append((n,ins[0],k,'fab',2))
        for n in self.l['osats']:
            for nn,k in self.pairs:
                if nn!=n or 'raw' not in self.names[k]: continue
                name=self.names[k].replace('_raw','')
                if name in self.names:
                    out=self.names.index(name)
                    if (n,out) in self.idx: self.production.append((n,k,out,'osat',1))
        self.fuel={}
        initial=self.s.get('instance',{}).get('initial_state',{})
        pipe=initial.get('pipeline',[]) if isinstance(initial,dict) else []
        if isinstance(pipe,list):
            for x in pipe:
                if not isinstance(x,dict): continue
                try:
                    e=x.get('edge'); k=x.get('k',x.get('commodity'))
                    if isinstance(e,str): e=self.s['edges']['id'].index(e)
                    if isinstance(k,str): k=self.names.index(k)
                    e=int(e); k=int(k); n=self.head[e]
                    lane=x.get('lane')
                    if isinstance(lane,str): lane=self.s['lanes']['id'].index(lane)
                    if lane is not None and int(lane)>=0: n=self.head[self.s['lanes']['edges'][int(lane)][-1]]
                    if n in self.l['grids']:
                        self.fuel[n,k]=self.fuel.get((n,k),0)+float(x.get('qty',x.get('quantity',0)))/max(1,int(self.s['edges']['tau0'][e]))
                except (ValueError,TypeError,IndexError): pass
        self.first=True

    def act(self,o):
        week=int(o['week'][0]); H=min(14,self.T-week+1); P=len(self.pairs); A=len(self.routes)
        def read(key,default):
            x=np.asarray(o.get(key,default)); m=np.asarray(o.get(key+'.observed',np.ones_like(x)))
            return np.where(m,x,default)
        cap=read('graph_now.u',np.array([0 if x is None else x for x in self.s['edges']['u0']],float))
        tau=read('graph_now.tau',np.array(self.s['edges']['tau0']))
        cost=read('graph_now.c',np.array(self.s['edges']['c0']))
        tariff=read('graph_now.tariff',np.zeros((len(cap),len(self.names))))
        opened=read('graph_now.open',np.ones(len(self.cp)))
        initial=np.zeros(P)
        for p,q in zip(self.stock,read('stock.qty',np.zeros(len(self.stock)))): initial[self.idx[p]]=q
        if self.first:
            for n,k in self.pairs:
                if n in self.l['grids'] and self.names[k] in ('lng','crude','nucfuel'):
                    self.fuel.setdefault((n,k),initial[self.idx[n,k]]/5)
            self.first=False
        inc=np.zeros((H,P))
        for j in np.flatnonzero(o.get('pipeline.qty.observed',np.zeros_like(o['pipeline.qty']))):
            e=int(o['pipeline.edge'][j]); k=int(o['pipeline.k'][j]); aw=int(o['pipeline.arrival_week'][j]); node=self.head[e]
            lane=int(o['pipeline.lane'][j]) if o.get('pipeline.lane.observed',np.zeros_like(o['pipeline.qty']))[j] else -1
            if 0<=lane<len(self.s['lanes']['edges']):
                es=self.s['lanes']['edges'][lane]
                if e in es:
                    aw+=sum(max(1,int(tau[z])) for z in es[es.index(e)+1:]); node=self.head[es[-1]]
            h=max(0,aw-week)
            if h<H and (node,k) in self.idx: inc[h,self.idx[node,k]]+=o['pipeline.qty'][j]
        for j in np.flatnonzero(o.get('wip.qty.observed',np.zeros_like(o.get('wip.qty',[])))):
            p=(int(o['wip.node'][j]),int(o['wip.k'][j])); h=max(0,int(o['wip.out_week'][j])-week)
            if h<H and p in self.idx: inc[h,self.idx[p]]+=o['wip.qty'][j]
        for key,row in zip(self.l.get('lot_keys',[]),o.get('queue_lots.qty',[])):
            n,k,l,e=key; es=[e]
            if l is not None and int(l)>=0:
                route=self.s['lanes']['edges'][int(l)]
                if e in route: es=route[route.index(e):]
            delay=sum(max(1,int(tau[z])) for z in es)
            if opened[self.cp[n]]<=0: delay+=4
            p=(self.head[es[-1]],k)
            if delay<H and p in self.idx: inc[delay,self.idx[p]]+=np.sum(row)
        for p,q in zip(self.l['supply_slots'],read('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))):
            inc[:,self.idx[tuple(p)]]+=q
        forecast=read('demand_forecast.qty',np.zeros_like(o['demand_forecast.qty']))
        pi={ (n,k):float(v) for n,k,v in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['pi'])}
        backflag={(n,k):bool(v) for n,k,v in zip(self.s['sinks']['node'],self.s['sinks']['k'],self.s['sinks']['backlog'])}
        value={p:pi.get(p,0) for p in self.pairs}
        for _ in range(len(self.pairs)):
            old=value.copy()
            for a,b,es,k in self.routes: value[a,k]=max(value[a,k],.97*old[b,k])
            for n,ik,ok,typ,d in self.production: value[n,ik]=max(value[n,ik],.8*old[n,ok])
            if value==old: break
        c=[]; bounds=[]; er=[]; ec=[]; ev=[]; br=[]; bc=[]; bv=[]; rhs=[]
        balances=[[{} for _ in range(P)] for _ in range(H)]
        def var(price,upper=None):
            j=len(c); c.append(float(price)); bounds.append((0,upper)); return j
        def add(h,p,j,v): balances[h][p][j]=balances[h][p].get(j,0)+v
        def ub(items,limit):
            r=len(rhs); rhs.append(float(limit))
            for j,v in items: br.append(r); bc.append(j); bv.append(v)
        inv=np.empty((H,P),int)
        for h in range(H):
            for p,pair in enumerate(self.pairs):
                j=var(.01); inv[h,p]=j; add(h,p,j,1)
                if h: add(h,p,inv[h-1,p],-1)
        # Modest terminal inventory value, bounded to useful quantities.
        if week+H-1<self.T:
            for p,pair in enumerate(self.pairs):
                target=0
                for d,dp in enumerate(self.dem):
                    if dp[1]==pair[1]: target+=2*np.mean(forecast[d])
                if pair in self.fuel: target=max(target,2*self.fuel[pair])
                if target>0 and value[pair]>0:
                    j=var(-.25*value[pair],target); ub([(j,1),(inv[-1,p],-1)],0)
        pending={}
        for j in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            key=(int(o['pending_prohibitions.edge'][j]),int(o['pending_prohibitions.k'][j]))
            pending[key]=min(pending.get(key,self.T+1),int(o['pending_prohibitions.effective_week'][j]))
        resources={}; flow=np.full((H,A),-1,int); mask=o.get('action_mask',np.ones(A))
        for h in range(H):
            for a,(src,dst,es,k) in enumerate(self.routes):
                delay=sum(max(1,int(tau[e])) for e in es)
                if h+delay>=H or not mask[a]: continue
                if any(pending.get((e,k),self.T+1)<=week+h for e in es): continue
                cps=set(self.tail[e] for e in es if self.tail[e] in self.cp)
                if any(opened[self.cp[n]]<=0 for n in cps): continue
                price=sum(cost[e]+tariff[e,k]*self.s['commodities']['v'][k] for e in es)
                j=var(price,max(0,min(cap[e] for e in es))); flow[h,a]=j
                add(h,self.idx[src,k],j,1); add(h+delay,self.idx[dst,k],j,-1)
                for e in es: resources.setdefault((h,'edge',e),[]).append(j)
                for n in cps: resources.setdefault((h,self.s['commodities']['pool'][k],n),[]).append(j)
            prodgroups={}
            for n,ik,ok,typ,delay in self.production:
                if h+delay>=H: continue
                j=var(.01); add(h,self.idx[n,ik],j,1); add(h+delay,self.idx[n,ok],j,-1)
                prodgroups.setdefault((n,typ),[]).append(j)
            for (n,typ),js in prodgroups.items():
                nodes=self.l['fabs' if typ=='fab' else 'osats']; key='graph_now.fab.cap_eff' if typ=='fab' else 'graph_now.osat.thr_eff'
                caps=read(key,np.zeros(len(nodes))); ub([(j,1) for j in js],max(0,caps[nodes.index(n)]))
            for pair,rate in self.fuel.items():
                j=var(-3000,max(0,rate)); add(h,self.idx[pair],j,1)
        for (h,typ,n),js in resources.items():
            limit=cap[n] if typ=='edge' else read('graph_now.kappa.'+typ,np.full(len(self.cp),1e12))[self.cp[n]]*opened[self.cp[n]]
            ub([(j,1) for j in js],max(0,limit))
        backlog=read('backlog.qty',np.zeros(len(self.dem)))
        for d,pair in enumerate(self.dem):
            prev=None
            for h in range(H):
                qty=max(0,float(forecast[d,min(h,forecast.shape[1]-1)]))
                if backflag.get(pair,False):
                    unserved=var(pi.get(pair,1000)); served=var(0)
                    r=H*P+len(rhs) # separate equalities below through auxiliary list
                    # served + backlog_next - backlog_previous = new demand
                    if not hasattr(self,'dummy'): pass
                    ub([(served,1),(unserved,1)]+([] if prev is None else [(prev,-1)]),qty+(backlog[d] if h==0 else 0))
                    ub([(served,-1),(unserved,-1)]+([] if prev is None else [(prev,1)]),-qty-(backlog[d] if h==0 else 0))
                    prev=unserved
                else: served=var(-pi.get(pair,1000),qty)
                add(h,self.idx[pair],served,1)
        beq=[]
        for h in range(H):
            for p in range(P):
                r=len(beq); beq.append(inc[h,p]+(initial[p] if h==0 else 0))
                for j,v in balances[h][p].items(): er.append(r); ec.append(j); ev.append(v)
        N=len(c)
        result=linprog(c,A_ub=coo_matrix((bv,(br,bc)),shape=(len(rhs),N)).tocsr(),b_ub=rhs,A_eq=coo_matrix((ev,(er,ec)),shape=(H*P,N)).tocsr(),b_eq=beq,bounds=bounds,method='highs',options={'time_limit':15})
        out=np.zeros(A)
        if result.x is not None:
            for a,j in enumerate(flow[0]):
                if j>=0: out[a]=max(0,result.x[j])
        return {'flows':out}
