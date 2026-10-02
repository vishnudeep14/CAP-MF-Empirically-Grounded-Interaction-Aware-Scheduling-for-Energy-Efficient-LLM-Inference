from __future__ import annotations
import argparse,csv,json,random,statistics
from pathlib import Path
from scheduler.empirical_profiles import EmpiricalProfiles
from scheduler.models import Request
from scheduler.solvers import solve_exact,solve_greedy,solve_meanfield

def load_requests(path,limit=None):
    rows=list(csv.DictReader(open(path,encoding='utf-8')));rows=rows[:limit] if limit else rows
    return [Request(i,r['bucket']) for i,r in enumerate(rows)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--configs',default='data/processed/config_profiles.csv');p.add_argument('--interactions',default='data/processed/interactions.csv');p.add_argument('--requests',required=True);p.add_argument('--limit',type=int,default=12);p.add_argument('--lambda-interaction',type=float,default=1.0);p.add_argument('--tau',type=float,default=1.0);p.add_argument('--damping',type=float,default=.10);p.add_argument('--output',default='results/solver_results.csv');a=p.parse_args()
 profile=EmpiricalProfiles(a.configs,a.interactions);requests=load_requests(a.requests,a.limit);rows=[]
 solvers=[('exact',lambda:solve_exact(requests,profile,a.lambda_interaction)),('greedy',lambda:solve_greedy(requests,profile,a.lambda_interaction)),('meanfield',lambda:solve_meanfield(requests,profile,a.lambda_interaction,a.tau,a.damping))]
 exact=None
 for name,fn in solvers:
  assignment,obj,t,diag=fn();exact=obj if name=='exact' else exact
  rows.append({'solver':name,'objective':obj,'runtime_s':t,'assignment_json':json.dumps(assignment,sort_keys=True),**diag})
 for r in rows:r['gap_vs_exact_pct']=100*(r['objective']-exact)/abs(exact) if exact else 0
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for r in rows:print(r)
if __name__=='__main__':main()
