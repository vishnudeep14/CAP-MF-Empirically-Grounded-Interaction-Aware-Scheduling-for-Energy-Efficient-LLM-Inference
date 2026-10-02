from __future__ import annotations
from collections import Counter
from .models import Request

def validate_requests(requests, profile):
    d=[k for k,v in Counter(r.id for r in requests).items() if v>1]
    if d: raise ValueError(f'duplicate ids: {d}')
    bad=sorted({r.bucket for r in requests if r.bucket not in profile.buckets})
    if bad: raise ValueError(f'unknown buckets: {bad}')

def validate_assignment(requests,assignment,profile,hard_capacity=True):
    validate_requests(requests,profile)
    if {r.id for r in requests} != set(assignment): raise ValueError('assignment id mismatch')
    configs={c.name:c for c in profile.configs}; req={r.id:r for r in requests}; occ={c.name:0 for c in profile.configs}
    for rid,name in assignment.items():
        if name not in configs: raise ValueError(f'unknown config {name}')
        c=configs[name]
        if not profile.feasible(c,req[rid].bucket): raise ValueError(f'infeasible {rid}->{name}')
        occ[name]+=1
    if hard_capacity:
        bad={c.name:(occ[c.name],c.capacity) for c in profile.configs if occ[c.name]>c.capacity}
        if bad: raise ValueError(f'capacity exceeded: {bad}')
    return occ
