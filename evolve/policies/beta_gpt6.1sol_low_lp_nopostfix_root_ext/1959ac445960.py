# 0.6828211219609539
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


class Agent:
    def __init__(self, config):
        self.c = config
        self.s = config['static']; self.l = config['layout']
        self.raw = self.s['instance']; self.nodes = self.raw['nodes']
        self.ed = self.s['edges']; self.la = self.s['lanes']
        self.kids = self.s['commodities']['id']; self.ki = {k:i for i,k in enumerate(self.kids)}
        self.ni = {n:i for i,n in enumerate(self.s['nodes']['id'])}
        self.slots = [tuple(x) for x in self.l['stock_slots']]
        self.si = {x:i for i,x in enumerate(self.slots)}
        self.cp = {n:i for i,n in enumerate(self.l['chokepoints'])}
        self.cache = {}; self.routes = []
        self.storage = []; self.hold = []; self.salvage = []
        for n,k in self.slots:
            d = self.nodes[n].get('stock',{}).get(self.kids[k],{})
            self.storage.append(float(d['storage']) if d.get('storage') is not None else 1e12)
            self.hold.append(float(d.get('holding_cost',0)))
            self.salvage.append(float(d.get('salvage',0)))
        def duration(ref):
            if not ref: return 0
            typ = 'lane' if 'lane' in ref else 'edge'; idx = ref[typ]
            tab = self.la if typ=='lane' else self.ed
            if isinstance(idx,str): idx = tab['id'].index(idx)
            return sum(self.ed['tau0'][e] for e in self.la['edges'][idx]) if typ=='lane' else self.ed['tau0'][idx]
        for j,e in enumerate(self.s['action_slots']['edge']):
            k = self.s['action_slots']['k'][j]; ln = self.s['action_slots']['lane'][j]
            es = list(self.la['edges'][ln]) if ln is not None and ln>=0 else [e]
            dt = 0
            if ln is not None and ln>=0 and self.la['alt_of'][ln]:
                dt += max(0,sum(self.ed['tau0'][v] for v in es)-duration(self.la['alt_of'][ln]))
            if self.ed['mode'][e]=='sea' and self.ed['alt_of'][e]:
                dt += max(0,self.ed['tau0'][e]-duration(self.ed['alt_of'][e]))
            self.routes.append((e,k,ln,es,self.ed['tail'][e],self.ed['head'][es[-1]],dt))

    def act(self,o):
        def get(key,default):
            a = np.asarray(o.get(key,default),dtype=float); m = o.get(key+'.observed')
            if m is not None and np.shape(m)==a.shape:
                a = np.where(m,a,self.cache.get(key,np.asarray(default,dtype=float)))
            self.cache[key] = a.copy(); return a
        week = int(o['week'][0]); rem = int(self.c['T'])-week+1; H = min(26,rem)
        S = len(self.slots); E = len(self.ed['id']); K = len(self.kids)
        stock = get('stock.qty',np.zeros(S))
        tau = np.maximum(1,get('graph_now.tau',self.ed['tau0'])).astype(int)
        normal = np.array([v if v is not None else 0 for v in self.ed['u0']],float)
        u = get('graph_now.u',normal); c = get('graph_now.c',self.ed['c0'])
        z = get('graph_now.prohibited',np.zeros((E,K)))
        tariff = get('graph_now.tariff',np.zeros((E,K)))
        opening = get('graph_now.open',np.ones(len(self.cp)))
        war = get('graph_now.war_risk',np.zeros(len(self.cp))).astype(int)
        kap = {p:get('graph_now.kappa.'+p,np.full(len(self.cp),1e12)) for p in ('tb','ct')}
        def blend(a,b,h,d=9): return a+max(0,b-a)*(1-np.exp(-max(0,h-2)/d))
        def cap(e,h): return max(0,blend(u[e],normal[e],h,7))
        ends = {}
        for i in np.flatnonzero(o.get('closure_end.end_week.observed',[])):
            if o['closure_end.chokepoint.observed'][i]:
                n = int(o['closure_end.chokepoint'][i]); ends[n] = max(ends.get(n,week),int(o['closure_end.end_week'][i]))
        def reopen(n): return max(0,ends.get(n,week+5)-week+1)
        pending = {}
        for i in np.flatnonzero(o.get('pending_prohibitions.edge.observed',[])):
            key = int(o['pending_prohibitions.edge'][i]),int(o['pending_prohibitions.k'][i])
            pending[key] = min(pending.get(key,10**9),int(o['pending_prohibitions.effective_week'][i]))
        def blocked(e,k,h): return z[e,k] or week+h>=pending.get((e,k),10**9)
        def capacity(n,p,h):
            i = self.cp[n]; a = self.nodes[n]['chokepoint']; mu = a.get('mu',{})
            norm = float(a.get('k_c',1))*float(mu.get(p,kap[p][i])) if isinstance(mu,dict) else kap[p][i]
            if opening[i]<.1: return max(0,norm if h>=reopen(n) else kap[p][i])
            return max(0,blend(kap[p][i],norm,h,7))
        def suffix(e,ln):
            if ln is not None and ln>=0:
                es = self.la['edges'][int(ln)]
                if e in es: return list(es[es.index(e)+1:])
            return []
        arrivals = np.zeros((S,H)); queues = {n:[] for n in self.cp}; scheduled = [[] for _ in range(H)]
        def arrive(n,k,q,h):
            s = self.si.get((int(n),int(k)))
            if s is not None and 0<=h<H: arrivals[s,int(h)] += q
        for i in np.flatnonzero(o.get('pipeline.qty.observed',[])):
            e = int(o['pipeline.edge'][i]); k = int(o['pipeline.k'][i]); q = float(o['pipeline.qty'][i])
            ln = int(o['pipeline.lane'][i]) if o.get('pipeline.lane.observed',np.zeros(len(o['pipeline.qty'])))[i] else None
            h = int(o['pipeline.arrival_week'][i])-week; n = self.ed['head'][e]; rest = suffix(e,ln)
            if n in self.cp and rest:
                if 0<=h<H: scheduled[h].append((n,[k,q,rest,h]))
            else: arrive(n,k,q,h)
        for i,(n,k,ln,e) in enumerate(self.l.get('lot_keys',[])):
            if 'queue_lots.qty' in o:
                for a in np.flatnonzero(o['queue_lots.qty'][i]>0):
                    queues[n].append([k,float(o['queue_lots.qty'][i,a]),[e]+suffix(e,ln),int(a)+1-week])
        ue = {}; up = {}
        for h in range(H):
            for n,lot in scheduled[h]: queues[n].append(lot)
            ec = np.array([cap(e,h) for e in range(E)])
            for n in self.cp:
                pc = {p:capacity(n,p,h) for p in kap}
                for lot in sorted(queues[n],key=lambda v:v[3]):
                    k,q,rest,age = lot; e = rest[0]; p = self.s['commodities']['pool'][k]
                    if blocked(e,k,h): continue
                    take = min(q,ec[e],pc[p])
                    if take<=0: continue
                    lot[1] -= take; ec[e] -= take; pc[p] -= take
                    ue[e,h] = ue.get((e,h),0)+take; up[n,p,h] = up.get((n,p,h),0)+take
                    hh = h+tau[e]; dst = self.ed['head'][e]
                    if dst in self.cp and len(rest)>1:
                        if hh<H: scheduled[hh].append((dst,[k,take,rest[1:],hh]))
                    else: arrive(dst,k,take,hh)
                queues[n] = [v for v in queues[n] if v[1]>1e-9]
        for i in np.flatnonzero(o.get('wip.qty.observed',[])):
            arrive(o['wip.node'][i],o['wip.k'][i],o['wip.qty'][i],int(o['wip.out_week'][i])-week)
        costs=[]; bounds=[]; eq=[]; rhs=[]; ub=[]; br=[]
        def var(cost=0,limit=None):
            j=len(costs); costs.append(float(cost)/100000); bounds.append((0,limit)); return j
        def row(rows,bs,terms,b): rows.append(terms); bs.append(float(b))
        I=[[var(self.hold[s]-(self.salvage[s] if rem==H and h==H-1 else 0),self.storage[s]) for h in range(H)] for s in range(S)]
        bal=[[{I[s][h]:1} for h in range(H)] for s in range(S)]; draws=[[{} for h in range(H)] for s in range(S)]
        def term(s,h,x,v):
            if s is not None and 0<=h<H: bal[s][h][x]=bal[s][h].get(x,0)+v
        def buffer(s,h,target,price):
            d=var(price); row(ub,br,{I[s][h]:-1,d:-1},-max(0,min(target,.9*self.storage[s])))
        for s,(n,k) in enumerate(self.slots):
            for h in range(H): term(s,h,var(self.raw['commodities'][k].get('disposal_cost',3000)),1)
        X=[]; eu={}; pu={}; fu={}; mask=o.get('action_mask',np.ones(len(self.routes)))
        for j,(e,k,ln,es,src,dst,dt) in enumerate(self.routes):
            a=self.si.get((src,k)); b=self.si.get((dst,k)); p=self.s['commodities']['pool'][k]
            price=sum(c[v]+tariff[v,k]*self.s['commodities']['v'][k] for v in es)
            for v in es:
                n=self.ed['tail'][v]
                if n in self.cp:
                    w=min(2,max(0,war[self.cp[n]])); charges=self.nodes[n]['chokepoint'].get('war_risk_cost',{}).get(self.kids[k],{})
                    if isinstance(charges,dict): price+=float(charges.get(('none','red_sea','hormuz_2026')[w],0))
                    elif isinstance(charges,(tuple,list)): price+=float(charges[w])
            xs=[]
            for h in range(H):
                hh=h; visits=[]; invalid=False
                for edge in es:
                    n=self.ed['tail'][edge]
                    if n in self.cp:
                        if opening[self.cp[n]]<.1: hh=max(hh,reopen(n))
                        while hh<H and (cap(edge,hh)-ue.get((edge,hh),0)<=1e-8 or capacity(n,p,hh)-up.get((n,p,hh),0)<=1e-8): hh+=1
                    if blocked(edge,k,hh): invalid=True
                    visits.append((edge,hh)); hh+=tau[edge]
                limit=cap(e,h); priceh=price+.01*(hh-h)
                if hh>=H:
                    if rem==H and len(es)==1 and b is not None: priceh-=self.salvage[b]
                    else: limit=0
                if invalid or a is None or b is None or (h==0 and not mask[j]): limit=0
                x=var(priceh,limit); xs.append(x); term(a,h,x,1); term(b,hh,x,-1)
                if a is not None: draws[a][h][x]=1
                if dt>0: fu.setdefault((p,h),{})[x]=dt
                for edge,t in visits:
                    if t<H:
                        eu.setdefault((edge,t),{})[x]=1; n=self.ed['tail'][edge]
                        if n in self.cp: pu.setdefault((n,p,t),{})[x]=1
            X.append(xs)
        for (e,h),terms in eu.items(): row(ub,br,terms,max(0,cap(e,h)-ue.get((e,h),0)))
        for (n,p,h),terms in pu.items(): row(ub,br,terms,max(0,capacity(n,p,h)-up.get((n,p,h),0)))
        params=self.raw['params']
        for (p,h),terms in fu.items():
            sh=params.get('fleet_share'); me=params.get('fleet_measure'); i=('tb','ct').index(p)
            if sh is not None and me is not None: row(ub,br,terms,float(sh[p] if isinstance(sh,dict) else sh[i])*float(me[p] if isinstance(me,dict) else me[i]))
        supply=get('graph_now.supply.avail',np.zeros(len(self.l['supply_slots'])))
        for i,(n,k) in enumerate(self.l['supply_slots']):
            s=self.si.get((n,k)); norm=self.nodes[n]['stock'][self.kids[k]].get('supply_rate',supply[i])
            for h in range(H): term(s,h,var(0,max(0,blend(supply[i],float(norm),h))),-1)
        R=get('graph_now.fab.R',np.ones(len(self.l['fabs'])))
        alpha=get('graph_now.fab.alpha_bar',np.ones(len(R)))
        G=get('graph_now.grid.G_bar',[self.nodes[n]['grid']['deliverable'] for n in self.l['grids']])
        Y=get('graph_now.grid.y_bar',[self.nodes[n]['grid']['base_load'] for n in self.l['grids']])
        energy={}; rates={}; targets={}
        for gi,n in enumerate(self.l['grids']):
            g=self.nodes[n]['grid']; load=float(Y[gi])
            for fi,fn in enumerate(self.l['fabs']):
                f=self.nodes[fn]['fab']
                if f.get('grid')==self.s['nodes']['id'][n] and R[fi]>0: load+=float(f['e'])*float(f['cap0'])*alpha[fi]
            frac=min(1,load/max(G[gi],1e-12))
            energy[n]=[]
            for h in range(H):
                shed=var(g.get('voll',4100000),max(0,Y[gi])); energy[n].append(({shed:-1},float(g['shares'].get('unmodelled',0))*G[gi]-Y[gi]))
            for fuel,share in g['shares'].items():
                if fuel not in self.ki: continue
                s=self.si[n,self.ki[fuel]]; rate=float(share)*G[gi]*frac; rates[s]=rate
                ibar=float(g.get('ibar',{}).get(fuel,0)); threshold=float(params.get('psi',0))*ibar
                targets[s]=max(ibar,rate*5)
                for h in range(H):
                    burn=var(.1,max(0,float(share)*G[gi])); term(s,h,burn,1); energy[n][h][0][burn]=-1
                    if fuel==g.get('rationed') and threshold>0:
                        mx=float(share)*G[gi]/threshold
                        row(ub,br,{burn:1} if h==0 else {burn:1,I[s][h-1]:-mx},max(0,mx*stock[s]) if h==0 else 0)
                    if rem-h>1:
                        buffer(s,h,min(max(rate*3.2,threshold*1.2 if fuel==g.get('rationed') else 0),rate*(rem-h-1)),24000)
                        buffer(s,h,min(rate*5.5,rate*(rem-h-1)),3000)
        for s,(n,k) in enumerate(self.slots):
            if self.nodes[n]['type']=='terminal':
                dests={dst for e,kk,ln,es,src,dst,dt in self.routes if src==n and kk==k}
                rate=sum(rates.get(self.si.get((d,k)),0) for d in dests)
                for h in range(H):
                    if rate>0 and rem-h>3: buffer(s,h,min(rate*2,rate*(rem-h-3)),3200)
        for fi,n in enumerate(self.l['fabs']):
            f=self.nodes[n]['fab']; a=self.si[n,self.ki[f['input']]]; b=self.si[n,self.ki[f['product']]]; grid=self.ni.get(f.get('grid'))
            targets[a]=min(.8*self.storage[a],float(f['cap0'])*2)
            for h in range(H):
                restoration=blend(R[fi],1,h,10); capacity=float(f['cap0'])*restoration*alpha[fi]
                credit=self.salvage[a] if rem==H and h+int(f['tau'])>=H else 0
                x=var(-credit,max(0,capacity)); term(a,h,x,1); term(b,h+int(f['tau']),x,-1)
                if grid in energy and restoration>0: energy[grid][h][0][x]=float(f['e'])/restoration
                if rem-h>int(f['tau'])+5: buffer(a,h,capacity,700)
        for n in energy:
            for terms,b in energy[n]: row(ub,br,terms,b)
        thr=get('graph_now.osat.thr_eff',[self.nodes[n]['osat']['thr'] for n in self.l['osats']])
        for oi,n in enumerate(self.l['osats']):
            os=self.nodes[n]['osat']
            for h in range(H):
                terms={}; capacity=blend(thr[oi],float(os['thr']),h,10)
                for raw,pk in os['packages'].items():
                    a=self.si[n,self.ki[raw]]; b=self.si[n,self.ki[pk]]
                    credit=self.salvage[a] if rem==H and h+int(os['tau'])>=H else 0
                    x=var(-credit); terms[x]=1; term(a,h,x,1); term(b,h+int(os['tau']),x,-1)
                    if rem-h>int(os['tau'])+4: buffer(a,h,capacity*.4/max(1,len(os['packages'])),450)
                row(ub,br,terms,max(0,capacity))
        forecast=get('demand_forecast.qty',np.zeros((len(self.l['demands']),1)))
        backlog=get('backlog.qty',np.zeros(len(self.l['demands'])))
        for di,(n,k) in enumerate(self.l['demands']):
            s=self.si[n,k]; back=self.s['sinks']['backlog'][di]; prev=None; pi=float(self.s['sinks']['pi'][di])
            nominal=float(self.nodes[n].get('sink',{}).get('demand',{}).get(self.kids[k],{}).get('dbar',np.mean(forecast[di])))
            for h in range(H):
                v=forecast[di,min(h,forecast.shape[1]-1)]; weight=np.exp(-max(0,h-forecast.shape[1]+1)/9)
                demand=max(0,weight*v+(1-weight)*nominal)
                served=var(0,None if back else demand); miss=var(pi); term(s,h,served,1); terms={served:1,miss:1}
                if back and prev is not None: terms[prev]=-1
                row(eq,rhs,terms,demand+(backlog[di] if back and h==0 else 0)); prev=miss
                if rem-h>2: buffer(s,h,demand*.8,pi*.16); buffer(s,h,demand*1.5,pi*.035)
            targets[s]=min(.8*self.storage[s],np.mean(forecast[di])*1.8)
        for s in range(S):
            for h in range(H):
                if h: bal[s][h][I[s][h-1]]=-1
                row(eq,rhs,bal[s][h],arrivals[s,h]+(stock[s] if h==0 else 0))
                terms=draws[s][h].copy()
                if h: terms[I[s][h-1]]=-1
                if terms: row(ub,br,terms,stock[s] if h==0 else 0)
            if rem>H and s in targets: buffer(s,H-1,targets[s],250000 if s in rates else 1500)
        def matrix(rows):
            rr=[]; cc=[]; vv=[]
            for i,terms in enumerate(rows):
                for j,v in terms.items():
                    if v: rr.append(i); cc.append(j); vv.append(v)
            return coo_matrix((vv,(rr,cc)),shape=(len(rows),len(costs))).tocsr()
        result=linprog(costs,A_ub=matrix(ub),b_ub=np.asarray(br),A_eq=matrix(eq),b_eq=np.asarray(rhs),bounds=bounds,method='highs',options={'time_limit':8})
        flows=np.zeros(len(self.routes))
        if result.x is not None and np.all(np.isfinite(result.x)):
            for j in range(len(flows)): flows[j]=max(0,float(result.x[X[j][0]]))
        return {'flows':flows}
