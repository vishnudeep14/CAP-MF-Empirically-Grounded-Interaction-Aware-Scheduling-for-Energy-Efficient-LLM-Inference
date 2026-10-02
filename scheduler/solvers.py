from __future__ import annotations
import math,time
from .canonical import validate_assignment,validate_requests


def total_objective(requests,assignment,profile,lam):
    cfg={c.name:c for c in profile.configs}; base=sum(profile.energy(cfg[assignment[r.id]],r.bucket) for r in requests)
    by={c.name:[] for c in profile.configs}
    for r in requests:by[assignment[r.id]].append(r)
    inter=0.0
    for c in profile.configs:
        rs=by[c.name]
        for a in range(len(rs)):
            for b in range(a+1,len(rs)):
                # Symmetric pair cost built from directed measured slowdowns.
                inter += .5*(profile.interaction(c,rs[a].bucket,rs[b].bucket)+profile.interaction(c,rs[b].bucket,rs[a].bucket))
    return base+lam*inter,base,inter


def solve_exact(requests,profile,lam=1.0):
    """Exact interaction-aware ILP with pairwise linearization."""
    import pulp
    t=time.perf_counter();validate_requests(requests,profile)
    opts={r.id:profile.feasible_configs(r.bucket) for r in requests}
    if any(not x for x in opts.values()):raise ValueError('request has no measured feasible config')
    prob=pulp.LpProblem('empirical_interaction_assignment',pulp.LpMinimize)
    x={(r.id,c.name):pulp.LpVariable(f'x_{r.id}_{c.name}',cat='Binary') for r in requests for c in opts[r.id]}
    components=[x[(r.id,c.name)]*profile.energy(c,r.bucket) for r in requests for c in opts[r.id]]
    y={}
    for ai in range(len(requests)):
        for bi in range(ai+1,len(requests)):
            ra,rb=requests[ai],requests[bi]
            common={c.name:c for c in opts[ra.id] if c.name in {z.name for z in opts[rb.id]}}
            for name,c in common.items():
                var=pulp.LpVariable(f'y_{ra.id}_{rb.id}_{name}',cat='Binary');y[(ra.id,rb.id,name)]=var
                prob += var <= x[(ra.id,name)]; prob += var <= x[(rb.id,name)]
                prob += var >= x[(ra.id,name)]+x[(rb.id,name)]-1
                j=.5*(profile.interaction(c,ra.bucket,rb.bucket)+profile.interaction(c,rb.bucket,ra.bucket))
                components.append(lam*j*var)
    prob += pulp.lpSum(components)
    for r in requests:prob += pulp.lpSum(x[(r.id,c.name)] for c in opts[r.id])==1
    for c in profile.configs:
        vs=[x[(r.id,c.name)] for r in requests if (r.id,c.name) in x]
        if vs:prob += pulp.lpSum(vs)<=c.capacity
    status=prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status]!='Optimal':raise RuntimeError(f'exact status={pulp.LpStatus[status]}')
    assignment={}
    for r in requests:
        picked=[c.name for c in opts[r.id] if (pulp.value(x[(r.id,c.name)]) or 0)>0.5]
        if len(picked)!=1:raise RuntimeError(f'invalid exact decision {r.id}')
        assignment[r.id]=picked[0]
    validate_assignment(requests,assignment,profile,True)
    obj,base,inter=total_objective(requests,assignment,profile,lam)
    return assignment,obj,time.perf_counter()-t,{'base_energy':base,'interaction':inter}


def solve_greedy(requests,profile,lam=1.0):
    t=time.perf_counter();validate_requests(requests,profile);rem={c.name:c.capacity for c in profile.configs};a={};cfg={c.name:c for c in profile.configs}
    # Hardest measured solo latency first.
    order=sorted(requests,key=lambda r:-min(profile.latency(c,r.bucket) for c in profile.feasible_configs(r.bucket)))
    for r in order:
        candidates=[c for c in profile.feasible_configs(r.bucket) if rem[c.name]>0]
        if not candidates:raise RuntimeError(f'no capacity for {r.id}')
        def incremental(c):
            pair=sum(.5*(profile.interaction(c,r.bucket,rr.bucket)+profile.interaction(c,rr.bucket,r.bucket)) for rr in requests if rr.id in a and a[rr.id]==c.name)
            return profile.energy(c,r.bucket)+lam*pair
        best=min(candidates,key=lambda c:(incremental(c),c.name));a[r.id]=best.name;rem[best.name]-=1
    validate_assignment(requests,a,profile,True);obj,base,inter=total_objective(requests,a,profile,lam)
    return a,obj,time.perf_counter()-t,{'base_energy':base,'interaction':inter}


def _softmin(vals,tau):
    if tau<=0:raise ValueError('tau must be >0')
    fin=[x for x in vals if math.isfinite(x)]
    if not fin:raise ValueError('no finite options')
    m=min(fin);w=[math.exp(-(x-m)/tau) if math.isfinite(x) else 0 for x in vals];z=sum(w);return [x/z for x in w]


def solve_meanfield(requests,profile,lam=1.0,tau=1.0,damping=.10,max_iterations=200,tol=1e-4):
    """Empirical interaction mean-field. J comes only from measured pair profiles."""
    t=time.perf_counter();validate_requests(requests,profile);configs=profile.configs;n=len(requests);m=len(configs)
    unary=[[profile.energy(c,r.bucket) if profile.feasible(c,r.bucket) else float('inf') for c in configs] for r in requests]
    q=[_softmin(row,tau) for row in unary];trace=[]
    for iteration in range(max_iterations):
        qn=[]
        for i,r in enumerate(requests):
            costs=[]
            for k,c in enumerate(configs):
                if not math.isfinite(unary[i][k]):costs.append(float('inf'));continue
                interaction=0.0
                for h,other in enumerate(requests):
                    if h==i:continue
                    j=.5*(profile.interaction(c,r.bucket,other.bucket)+profile.interaction(c,other.bucket,r.bucket))
                    interaction += q[h][k]*j
                # Smooth expected over-capacity discouragement; hard capacity is enforced at recovery.
                expected_occ=sum(q[h][k] for h in range(n))
                capacity_penalty=max(0.0,expected_occ-c.capacity)*max(profile.energy(c,r.bucket),1e-9)
                costs.append(unary[i][k]+lam*interaction+capacity_penalty)
            prop=_softmin(costs,tau);row=[damping*prop[k]+(1-damping)*q[i][k] for k in range(m)];z=sum(row);qn.append([x/z for x in row])
        residual=max(abs(qn[i][k]-q[i][k]) for i in range(n) for k in range(m)) if n else 0;trace.append(residual);q=qn
        if residual<tol:break
    # Confidence-ordered capacity-aware recovery using empirical total incremental cost.
    rem={c.name:c.capacity for c in configs};a={};order=sorted(range(n),key=lambda i:(-max(q[i]),requests[i].id))
    for i in order:
        r=requests[i];indices=[k for k,c in enumerate(configs) if profile.feasible(c,r.bucket) and rem[c.name]>0]
        if not indices:raise RuntimeError(f'no capacity for MF recovery request {r.id}')
        def key(k):
            c=configs[k];pair=sum(.5*(profile.interaction(c,r.bucket,rr.bucket)+profile.interaction(c,rr.bucket,r.bucket)) for rr in requests if rr.id in a and a[rr.id]==c.name)
            return (-q[i][k],profile.energy(c,r.bucket)+lam*pair,c.name)
        k=min(indices,key=key);a[r.id]=configs[k].name;rem[configs[k].name]-=1
    validate_assignment(requests,a,profile,True);obj,base,inter=total_objective(requests,a,profile,lam)
    return a,obj,time.perf_counter()-t,{'iterations':iteration+1,'converged':trace[-1]<tol if trace else True,'residual':trace[-1] if trace else 0,'base_energy':base,'interaction':inter,'simplex_error':max(abs(sum(row)-1) for row in q) if q else 0}
