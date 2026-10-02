"""Post-profiling reproducible empirical pipeline.
Profiling itself is intentionally separate because each real TP configuration
requires restarting/provisioning vLLM on the corresponding hardware.
"""
from __future__ import annotations
import argparse,subprocess,sys

def run(cmd):print('+',' '.join(cmd));subprocess.run(cmd,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--requests',default='data/processed/requests.csv');p.add_argument('--limit',type=int,default=12);a=p.parse_args()
 run([sys.executable,'validate.py'])
 run([sys.executable,'-m','experiments.run_solver_comparison','--requests',a.requests,'--limit',str(a.limit)])
 print('Post-profiling solver pipeline complete. Run empirical replay while the measured config endpoints are live.')
if __name__=='__main__':main()
