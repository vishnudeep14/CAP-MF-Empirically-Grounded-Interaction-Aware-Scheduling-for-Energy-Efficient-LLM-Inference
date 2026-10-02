"""Strict empirical profile loader. No synthetic fallback exists here."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
from .models import Config

class EmpiricalProfileError(ValueError): pass

class EmpiricalProfiles:
    def __init__(self, configs_csv: str, interactions_csv: str):
        self.config_rows = list(csv.DictReader(open(configs_csv, encoding='utf-8')))
        self.interaction_rows = list(csv.DictReader(open(interactions_csv, encoding='utf-8')))
        if not self.config_rows: raise EmpiricalProfileError('empty empirical config profile')
        self._cfg = {}
        for r in self.config_rows:
            key=(r['config'],r['bucket'])
            vals={k: float(r[k]) for k in ['latency_s','energy_j','slo_s']}
            if not all(math.isfinite(v) and v>0 for v in vals.values()): raise EmpiricalProfileError(f'invalid profile {key}')
            vals['tp']=int(r['tp']); vals['capacity']=int(r['capacity'])
            self._cfg[key]=vals
        self._j={}
        for r in self.interaction_rows:
            key=(r['config'],r['bucket_i'],r['bucket_j'])
            j=float(r['interaction'])
            if not math.isfinite(j): raise EmpiricalProfileError(f'invalid interaction {key}')
            self._j[key]=j
        names=sorted({r['config'] for r in self.config_rows})
        self.configs=[]
        for name in names:
            rs=[r for r in self.config_rows if r['config']==name]
            tps={int(r['tp']) for r in rs}; caps={int(r['capacity']) for r in rs}
            if len(tps)!=1 or len(caps)!=1: raise EmpiricalProfileError(f'inconsistent config metadata {name}')
            self.configs.append(Config(name,next(iter(tps)),next(iter(caps))))
        self.buckets=sorted({r['bucket'] for r in self.config_rows})
        expected={(c.name,b) for c in self.configs for b in self.buckets}
        missing=expected-set(self._cfg)
        if missing: raise EmpiricalProfileError(f'missing measured config/bucket cells: {sorted(missing)}')
        # Require complete directed J including diagonal for every config.
        missing_j=[]
        for c in self.configs:
            for bi in self.buckets:
                for bj in self.buckets:
                    if (c.name,bi,bj) not in self._j: missing_j.append((c.name,bi,bj))
        if missing_j: raise EmpiricalProfileError(f'missing measured interactions, first={missing_j[:5]}')

    def latency(self,c:Config,b:str)->float:return self._cfg[(c.name,b)]['latency_s']
    def energy(self,c:Config,b:str)->float:return self._cfg[(c.name,b)]['energy_j']
    def slo(self,b:str)->float:
        vals={self._cfg[(c.name,b)]['slo_s'] for c in self.configs}
        if len(vals)!=1: raise EmpiricalProfileError(f'inconsistent SLO for {b}')
        return next(iter(vals))
    def feasible(self,c,b)->bool:return self.latency(c,b)<=self.slo(b)
    def feasible_configs(self,b):return [c for c in self.configs if self.feasible(c,b)]
    def interaction(self,c,bi,bj)->float:return self._j[(c.name,bi,bj)]
    def to_metadata(self): return {'configs':[c.__dict__ for c in self.configs],'buckets':self.buckets}
