from __future__ import annotations
import argparse,math
from scheduler.empirical_profiles import EmpiricalProfiles
from scheduler.models import Request
from scheduler.canonical import validate_assignment
from scheduler.solvers import solve_exact,solve_greedy,solve_meanfield

def main():
 p=argparse.ArgumentParser();p.add_argument('--configs',default='data/processed/config_profiles.csv');p.add_argument('--interactions',default='data/processed/interactions.csv');a=p.parse_args();profile=EmpiricalProfiles(a.configs,a.interactions)
 print('[PASS] empirical profile complete and finite')
 rs=[Request(i,b) for i,b in enumerate(profile.buckets[:min(4,len(profile.buckets))])]
 # Skip if measured capacities cannot fit this validation workload.
 if sum(c.capacity for c in profile.configs)<len(rs):raise RuntimeError('measured capacities too small for validator')
 ea,eo,_,_=solve_exact(rs,profile);ga,go,_,_=solve_greedy(rs,profile);ma,mo,_,md=solve_meanfield(rs,profile)
 validate_assignment(rs,ea,profile);validate_assignment(rs,ga,profile);validate_assignment(rs,ma,profile)
 print('[PASS] Exact assignment valid');print('[PASS] Greedy assignment valid');print('[PASS] Mean-Field assignment valid')
 if eo>go+1e-8 or eo>mo+1e-8:raise RuntimeError(f'exact lower bound violated: {eo},{go},{mo}')
 print('[PASS] Exact <= Greedy/Mean-Field objective')
 if md['simplex_error']>1e-9:raise RuntimeError('MF simplex invalid')
 print('[PASS] Mean-Field simplex valid')
 print(f"[INFO] Mean-Field converged={md['converged']} residual={md['residual']}")
 print('All empirical checks passed.')
if __name__=='__main__':main()
